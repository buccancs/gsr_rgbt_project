import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Path to the forensic CSV file
csv_file = Path(__file__).parent / "project_git_history_forensic.csv"
# Output file path
output_file = Path(__file__).parent.parent / "docs" / "repository_development_timeline.md"

def process_git_history():
    """Process the git history CSV and create a timeline document."""
    print(f"Processing git history from {csv_file}...")
    
    # Dictionary to store commits by hash
    commits = defaultdict(lambda: {
        'files': set(),
        'insertions': 0,
        'deletions': 0,
        'file_details': []
    })
    
    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            commit_hash = row['commit_hash']
            
            # If this is the first time we're seeing this commit, store the metadata
            if 'author' not in commits[commit_hash]:
                commits[commit_hash].update({
                    'author': row['author_name'],
                    'date': datetime.strptime(row['author_date'], '%Y-%m-%d %H:%M:%S'),
                    'subject': row['commit_subject'],
                    'body': row['commit_body']
                })
            
            # Add file information
            file_path = row['file_path']
            commits[commit_hash]['files'].add(file_path)
            commits[commit_hash]['insertions'] += int(row['insertions'])
            commits[commit_hash]['deletions'] += int(row['deletions'])
            
            # Store detailed file changes
            commits[commit_hash]['file_details'].append({
                'path': file_path,
                'insertions': int(row['insertions']),
                'deletions': int(row['deletions'])
            })
    
    # Sort commits by date
    sorted_commits = sorted(commits.items(), key=lambda x: x[1]['date'])
    
    # Generate the timeline document
    generate_timeline(sorted_commits)

def generate_timeline(sorted_commits):
    """Generate a markdown timeline document from the processed commits."""
    print(f"Generating timeline document at {output_file}...")
    
    # Create the output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the header
        f.write("# GSR-RGBT Project Development Timeline\n\n")
        f.write("## Overview\n\n")
        f.write("This document provides a step-by-step timeline of the GSR-RGBT project's development, ")
        f.write("based on the repository's commit history. It shows the evolution of the project from ")
        f.write("initial scaffold to its current state, highlighting key milestones and changes.\n\n")
        
        # Group commits by month for better organization
        current_month = None
        
        for commit_hash, commit_data in sorted_commits:
            commit_date = commit_data['date']
            month_year = commit_date.strftime('%B %Y')
            
            # Add a new month header if we've moved to a new month
            if month_year != current_month:
                current_month = month_year
                f.write(f"\n## {month_year}\n\n")
            
            # Write commit information
            f.write(f"### {commit_date.strftime('%Y-%m-%d %H:%M:%S')} - {commit_data['subject']}\n\n")
            f.write(f"**Author:** {commit_data['author']}  \n")
            f.write(f"**Commit:** {commit_hash[:10]}  \n")
            f.write(f"**Changes:** {len(commit_data['files'])} files changed, ")
            f.write(f"{commit_data['insertions']} insertions(+), ")
            f.write(f"{commit_data['deletions']} deletions(-)  \n\n")
            
            # Write commit body if it exists
            if commit_data['body']:
                f.write("**Description:**\n\n")
                f.write(f"```\n{commit_data['body']}\n```\n\n")
            
            # Write file changes
            f.write("**Files Changed:**\n\n")
            
            # Group files by directory for better organization
            files_by_dir = defaultdict(list)
            for file_detail in commit_data['file_details']:
                file_path = file_detail['path']
                if file_path:  # Skip empty file paths
                    dir_path = os.path.dirname(file_path) or '.'
                    files_by_dir[dir_path].append(file_detail)
            
            # Write files grouped by directory
            for dir_path, files in sorted(files_by_dir.items()):
                f.write(f"* **{dir_path}/**\n")
                for file_detail in sorted(files, key=lambda x: x['path']):
                    file_name = os.path.basename(file_detail['path'])
                    f.write(f"  * {file_name} ")
                    f.write(f"(+{file_detail['insertions']}, -{file_detail['deletions']})\n")
            
            f.write("\n---\n\n")
    
    print(f"Timeline document generated successfully!")

if __name__ == "__main__":
    process_git_history()