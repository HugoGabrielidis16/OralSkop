# Git workflow — feature → dev → main

## Branch model

```
main        ← stable, deployable. Only receives merges from dev.
dev         ← integration branch. Feature branches merge here first.
feature/*   ← one branch per feature. Created off dev, merged back into dev.
```

Never commit directly to `main`. Never commit directly to `dev` (except hotfixes).

---

## The correct sequence — new feature

```bash
# 1. Start from dev, make sure it's up to date
git checkout dev
git pull origin dev

# 2. Create a feature branch off dev
git checkout -b feature/my-feature

# 3. Work. Stage and commit as you go.
git add <files>
git commit -m "feat(scope): short description"

# 4. When done — get any new dev changes before merging
git checkout dev
git pull origin dev
git checkout feature/my-feature
git rebase dev          # replay your commits on top of latest dev

# 5. Merge feature into dev (no fast-forward — keeps branch history visible)
git checkout dev
git merge --no-ff feature/my-feature -m "merge: feature/my-feature into dev"
git push origin dev

# 6. Delete the feature branch (it's done)
git branch -d feature/my-feature
git push origin --delete feature/my-feature

# 7. When dev is stable and ready to release — merge dev into main
git checkout main
git pull origin main
git merge --no-ff dev -m "release: backend scaffold"
git push origin main
```

---

## Commit message convention

```
feat(scope): what was added
fix(scope): what was fixed
chore: non-functional work (deps, config, gitignore)
refactor(scope): restructure without behaviour change
docs: documentation only
```

---

## What went wrong in this session (and why)

### Error 1 — `git reset HEAD .` before any commit

```
fatal: ambiguous argument 'HEAD': unknown revision or path not in the working tree.
```

**Why:** `HEAD` refers to the current commit. On a brand-new repo with no commits yet, `HEAD` doesn't exist. You can't reset to something that doesn't exist.

**Fix:** On a fresh repo, unstage with `git rm --cached <file>` instead. Or just commit first, then reset.

---

### Error 2 — first commit landed on `main` before branches existed

The correct order for a new repo is:
```bash
git init
git remote add origin <url>
git checkout -b main          # or let the first commit create it
# make initial commit here
git push -u origin main
git checkout -b dev           # branch off main
git checkout -b feature/xyz   # branch off dev
```

What happened instead: files were staged, commit was made on `main` by default, then branches were created — all pointing to the same commit. Not broken, just not clean.

Option A (what we did): accepted this state. Future branches will have the correct flow.

Option B (clean): would have been `git rebase -i --root` to restructure history, or starting over with an empty first commit on `main` before adding code.

---

### Error 3 — push to main rejected (non-fast-forward)

```
error: failed to push some refs to 'https://github.com/...'
hint: Updates were rejected because the remote contains work that you do not have locally.
```

**Why:** GitHub auto-created the repo with a default README commit. Our local `main` didn't have that commit, so Git refused to overwrite it — it would have silently discarded the README.

**Fix:** `git pull origin main --rebase` — fetches the remote commit, then replays our local commit on top of it. This produced a conflict in `.gitignore` because both sides had one.

**Conflict resolved by:** merging both files manually (remote had AI/ML patterns, local had backend patterns — kept both).

---

### The rebase vs merge choice on pull

`git pull --rebase` rewrites your local commit to sit *after* the remote commit in history.
`git pull` (merge) creates a merge commit.

For a feature branch being pulled into a shared branch, rebase keeps history linear and readable. For merging *between* branches (feature → dev → main), `--no-ff` merge is preferred — it preserves the fact that a feature was developed separately.