# Resolving merge conflicts: choosing current vs incoming changes

When Git shows a conflict it marks the two sides as **current change** (your branch) and **incoming change** (the branch you are merging/rebasing). You should rarely pick one blindly. Instead, choose based on intent:

- **Keep only your branch's code** → choose **current change**. Use this when the other branch's edits are obsolete or already handled.
- **Keep only the other branch's code** → choose **incoming change**. Use this when your local edits were superseded or were temporary/debug code.
- **Combine the useful parts** → manually edit to keep the best of both sides, then remove the conflict markers. This is the most common and safest approach.

## Practical steps
1. Read the conflict hunk and understand what each side represents.
2. Decide whether one side is clearly correct or whether they must be merged.
3. Edit the file to the desired final state (you can keep current, incoming, or a mix).
4. Remove all `<<<<<<<`, `=======`, and `>>>>>>>` markers.
5. Run tests or sanity checks, then `git add` and `git commit`.

## Tips
- Prefer **manual merges** when both sides add value; this prevents losing fixes.
- If unsure, search commit history or ask the author to understand intent.
- After resolving, run linters/tests to ensure the merged code still works.
