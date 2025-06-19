# src/scripts/generate_git_log.py

import csv
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s')


def check_git_repository():
    """Checks if the script is run within a Git repository."""
    result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.strip() != 'true':
        logging.error("This is not a Git repository. Please run this script from the project root.")
        return False
    return True


def get_commit_list() -> list:
    """Gets a list of all commit hashes in the repository, from oldest to newest."""
    command = ['git', 'rev-list', '--all', '--reverse']
    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
    return result.stdout.strip().split('\n')


def get_commit_details(commit_hash: str) -> dict:
    """Gets detailed metadata for a single commit."""
    field_separator = "<_FIELD_SEP_>"
    log_format = f"%an{field_separator}%ae{field_separator}%at{field_separator}%cn{field_separator}%ce{field_separator}%ct{field_separator}%s{field_separator}%b"
    command = ['git', 'show', '-s', f'--pretty=format:{log_format}', commit_hash]
    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

    parts = result.stdout.strip().split(field_separator, 7)
    author_name, author_email, author_date, committer_name, committer_email, committer_date, subject, body = parts

    return {
        'author_name': author_name, 'author_email': author_email,
        'author_date': datetime.fromtimestamp(int(author_date)).strftime('%Y-%m-%d %H:%M:%S'),
        'committer_name': committer_name, 'committer_email': committer_email,
        'committer_date': datetime.fromtimestamp(int(committer_date)).strftime('%Y-%m-%d %H:%M:%S'),
        'commit_subject': subject, 'commit_body': body.strip()
    }


def get_file_changes_with_diffs(commit_hash: str) -> list:
    """
    Gets file changes for a commit, including the raw diff content for each file.
    """
    # Use --patch to get the diff content along with numstat
    command = [
        'git', 'show', commit_hash,
        '--pretty=format:',  # Only want the changes, not the commit message
        '--numstat',
        '--patch'
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        logging.error(f"Git show command failed for commit {commit_hash[:7]}: {e}")
        return []

    # The output is complex: numstat lines, followed by full diff patches for each file.
    # We will parse it in stages.
    lines = result.stdout.strip().split('\n')
    changes = []
    current_file_path = None
    diff_content = []

    for line in lines:
        # Check if the line is a numstat line
        parts = line.split('\t')
        if len(parts) == 3 and (parts[0].isdigit() or parts[0] == '-') and (parts[1].isdigit() or parts[1] == '-'):
            insertions, deletions, file_path = parts
            changes.append({
                'file_path': file_path,
                'insertions': 0 if insertions == '-' else int(insertions),
                'deletions': 0 if deletions == '-' else int(deletions),
                'diff_content': ''  # Initialize diff content
            })
        # Check if the line indicates the start of a diff for a new file
        elif line.startswith('diff --git'):
            # This logic assumes the order of numstat and diffs is consistent
            file_path_from_diff = line.split(' b/')[-1]
            current_file_path = file_path_from_diff
            diff_content = [line]  # Start collecting diff for this file
        elif current_file_path:
            # Continue collecting diff lines for the current file
            diff_content.append(line)
            # Find the corresponding entry in our changes list and append the diff
            for change in changes:
                if change['file_path'] == current_file_path:
                    change['diff_content'] = '\n'.join(diff_content)
                    break

    return changes


def generate_forensic_git_log_data() -> list:
    """
    Parses the full git log to extract forensic-level detail about each commit.
    """
    if not check_git_repository(): return []

    logging.info("Starting forensic-level parsing of git log history...")
    log_data = []

    commit_hashes = get_commit_list()
    total_commits = len(commit_hashes)

    for i, commit_hash in enumerate(commit_hashes):
        logging.info(f"Processing commit {i + 1}/{total_commits} ({commit_hash[:7]}...)")
        commit_details = get_commit_details(commit_hash)
        file_changes = get_file_changes_with_diffs(commit_hash)

        if not file_changes:
            log_data.append({**commit_details, 'commit_hash': commit_hash})
        else:
            for change in file_changes:
                log_data.append({**commit_details, **change, 'commit_hash': commit_hash})

    logging.info(f"Successfully parsed data from {total_commits} commits.")
    return log_data


def write_log_to_csv(log_data: list, output_path: Path):
    """Writes the parsed git log data to a CSV file with robust quoting."""
    if not log_data:
        logging.warning("No log data to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        'commit_hash', 'author_name', 'author_email', 'author_date',
        'committer_name', 'committer_email', 'committer_date',
        'commit_subject', 'commit_body', 'file_path', 'insertions', 'deletions', 'diff_content'
    ]

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Use QUOTE_ALL to handle complex multi-line diff strings safely
            writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(log_data)
        logging.info(f"Forensic git history successfully written to {output_path}")
    except IOError as e:
        logging.error(f"Failed to write CSV file: {e}")


def main():
    """Main function to generate and save the git history log."""
    output_filename = "project_git_history_forensic.csv"
    output_dir = Path("")

    data = generate_forensic_git_log_data()
    write_log_to_csv(data, output_dir / output_filename)


if __name__ == "__main__":
    main()
