# Contributing to Eigenmind

Thanks for your interest in contributing! This project is open source, and we welcome any kind of contribution: bug fixes, new features, documentation improvements, refactoring, or simple suggestions. Every improvement, no matter how small, is appreciated.

## Open Source Basics

Open source thrives on collaboration and transparency. A few principles guide how we work together:

- **Discuss before coding.** Open an issue first so the change can be aligned with the project's direction before any work starts.
- **Work in the open.** Use issues and pull requests so anyone can follow along, comment, and learn.
- **Be respectful and constructive.** Reviews are about the code, not the person. Assume good intent on both sides.
- **Small, focused changes.** Smaller contributions are easier to review, test, and merge.

## Contribution Workflow

### 1. Open an issue

Every contribution starts with an issue. Whether it's a new feature, a bug, or another need, describe:

- **What** the problem or proposal is.
- **Why** it matters (context, use case, expected behavior).
- **How** you would approach it, if you already have an idea.

This gives maintainers and other contributors a chance to discuss the scope before any code is written.

### 2. Create a branch from the issue

Once the issue is agreed upon, create a dedicated branch from `main`. The branch name should follow the format `type/short-description` (see naming rules below).

```bash
git checkout main
git pull origin main
git checkout -b feat/add-sso-login
```

### 3. Develop on the branch

Make your changes on this branch only. Commit often with clear messages, and push regularly so progress is visible.

```bash
git add <files>
git commit -m "(feat:auth) Add SSO login form"
git push origin feat/add-sso-login
```

Make sure your code is tested locally and that existing tests still pass before moving on.

**Keep your branch up to date with `main`.** If your branch lives for more than a day or two, regularly rebase it on top of `main` so you work with the latest code and limit the size of future conflicts:

```bash
git checkout main
git pull origin main
git checkout feat/add-sso-login
git rebase main
git push --force-with-lease   # only needed if the branch was already pushed
```

**Handling conflicts.** If the rebase (or a merge) stops on a conflict, don't panic:

1. Run `git status` to see the conflicted files.
2. Open each one, look for the `<<<<<<<`, `=======`, `>>>>>>>` markers, and edit the file to keep the correct version.
3. Stage the fixed files and continue the rebase:

   ```bash
   git add <fixed-files>
   git rebase --continue
   ```

4. If you get lost or want to start over, you can always abort safely:

   ```bash
   git rebase --abort
   ```

### 4. Open a Pull Request

When the feature/fix is ready, open a Pull Request targeting `main`. In the PR description:

- Link the related issue (e.g. `Closes #42`).
- Summarize what changed and why.
- Mention how to test the change.

```bash
# Push your last changes, then open the PR via the GitHub UI
# or via the GitHub CLI:
gh pr create --base main --title "(feat:auth) Add SSO login" --body "Closes #42"
```

### 5. Wait for review from the code owners

The code owners listed in [CODEOWNERS](github/CODEOWNERS) will be automatically requested for review. Please be patient — reviewers will look at correctness, tests, style, and overall fit.

### 6. Iterate, then clean up

- **If the PR is approved and merged:** delete the branch so the repository stays clean.

  ```bash
  git checkout main
  git pull origin main
  git branch -d feat/add-sso-login
  git push origin --delete feat/add-sso-login
  ```

- **If changes are requested:** apply them on the same branch, push again, and re-request the review. Iterate with the reviewer until the PR is approved.

  You can either add new commits, or amend the last one to keep history clean during the back-and-forth:

  ```bash
  # Option A — add a new commit
  git add <files>
  git commit -m "(fix:auth) Handle empty token edge case"
  git push

  # Option B — amend the previous commit (rewrites history, force-push required)
  git add <files>
  git commit --amend --no-edit       # or --amend to edit the message
  git push --force-with-lease        # safer than --force: refuses if the remote moved
  ```

  Only force-push to **your own** feature branch, never to `main`. Use `--force-with-lease` rather than `--force` to avoid overwriting someone else's work.

### Squash before merging

All commits of a PR are **squashed into a single commit** when merged to `main`. This keeps the history of `main` linear and readable: one PR = one commit. You don't need to clean up your branch history manually — GitHub will do the squash at merge time — but make sure the **PR title** is clean, since it becomes the squash commit message.

## Basic Rules

A few rules everyone should follow:

### Naming conventions

We use two related formats — one for issues and commits, a simpler one for branches.

**Issues and commits** use the format:

```
(type:module) Short description
```

**Branches** use the format:

```
type/short-description
```

Where `type` is one of:

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding or updating tests
- `chore` — tooling, config, dependencies, etc.

Examples:

| Item    | Example                              |
| ------- | ------------------------------------ |
| Issue   | `(feat:auth) Add SSO login`          |
| Commit  | `(fix:ingest) Handle empty CSV rows` |
| Branch  | `feat/add-sso-login`                 |
| Branch  | `fix/handle-empty-csv-rows`          |
| Branch  | `docs/update-installation-steps`     |

### Pull Request quality

A PR must be **functional and tested** before requesting a merge:

- The code runs without errors.
- New behavior is covered by tests.
- Existing tests still pass.
- Lint/format checks pass.

If a PR is still a work in progress, mark it as a **draft** so reviewers know it's not ready yet.

### One issue, one branch, one PR

Create a **new issue and a new branch for each need**. Keep changes separated as much as possible:

- It makes reviews faster and more focused.
- It reduces the risk of conflicts.
- It keeps the git history readable.

If you find a second problem while working on something, open a separate issue for it instead of bundling it into the current PR.

---

Thanks again for contributing — happy coding!
