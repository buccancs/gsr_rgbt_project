# Guide to GitHub Workflows in the GSR-RGBT Project

## Introduction

This guide explains how to work with GitHub Actions workflows in the GSR-RGBT project. GitHub Actions is a continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline.

## Viewing Workflow Files in GitHub

If you can't see the workflow files in GitHub, follow these steps:

1. **Ensure the workflow files are committed and pushed to GitHub**:
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "Add CI workflow"
   git push origin main
   ```

2. **Navigate to the Actions tab in your GitHub repository**:
   - Go to your repository on GitHub (e.g., `https://github.com/username/gsr_rgbt_project`)
   - Click on the "Actions" tab in the top navigation bar
   - You should see your workflows listed here

3. **Check the file location**:
   - Workflow files must be in the `.github/workflows` directory
   - The file must have a `.yml` or `.yaml` extension
   - The file must be properly formatted YAML

## Understanding the CI Workflow

The GSR-RGBT project uses a CI workflow defined in `.github/workflows/ci.yml`. This workflow:

1. Runs on push to the main branch and on pull requests
2. Sets up a Python environment
3. Installs dependencies
4. Runs linting checks
5. Runs unit, smoke, and regression tests
6. Uploads coverage reports
7. Builds and uploads documentation

## Troubleshooting

If you're having issues with GitHub Actions workflows:

1. **Check workflow syntax**:
   - Ensure your YAML file is properly formatted
   - Use a YAML validator to check for syntax errors

2. **Check workflow logs**:
   - Click on a workflow run in the Actions tab
   - Examine the logs for any errors

3. **Check GitHub status**:
   - GitHub Actions might be experiencing issues
   - Check the [GitHub Status page](https://www.githubstatus.com/)

4. **Check repository permissions**:
   - Ensure GitHub Actions is enabled for your repository
   - Go to Settings > Actions > General and check the permissions

## Adding New Workflows

To add a new workflow:

1. Create a new YAML file in the `.github/workflows` directory
2. Define the workflow using the GitHub Actions syntax
3. Commit and push the file to GitHub

Example of a simple workflow:

```yaml
name: Simple Workflow

on:
  push:
    branches: [ main ]

jobs:
  hello:
    runs-on: ubuntu-latest
    steps:
    - name: Say Hello
      run: echo "Hello, World!"
```

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)