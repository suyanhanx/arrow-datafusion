<!--
Before opening a PR. Make sure following points are double-checked
-- Make sure `cargo fmt` is done.
-- Make sure `bash dev/rust_lint.sh` successfully runs.
-- Make sure `cargo test` passes from all tests.
-- Make sure there are wide-ranging unit tests, integration tests, end-to-end tests for the added functionality.
-- Make sure there is no merge conflict with target branch.
-- If there are sections in the code, that can be written as function do so
     Especially, if these sections are 
      - occur multiple times,
      - line number is more than > 20 approximately.
-- The code added doesn't contain unnecessary `.clone`s.
-- The code added doesn't contain naked `unwrap`s.
-- Make sure imports are consistent. Imports should be in following order
    - Standart library imports (alphabetically ordered within this group)
    - Arrow/ Datafusion imports (alphabetically ordered within this group)
    - #rd Party libraries (alphabetically ordered within this group)
    where each group is seperated by an empty line.
    
-- Inspect your attributes to what needs to be public, do not expose something if it is not used elsewhere.
-- Make sure that the docstrings of the newly added code blocks are not missing.
-- Read through comments whether they are clear, and grammatically correct.
-- Look for whether for loops can be re-written as iterator. Be pro-functional style.
-- Do not use short names such as `res`, `elem`, instead use `result`, `item`, etc.
-- For short and common functions use namespace with it to be explicit.
   (such as instead of `min`, `max`, use `std::cmp::min`, `std::cmp::max` etc.)
-- Add necessary license notice for the added code that will remain in the Synnada repo.
-- Make sure the PR body is understandable and summarizes the topic well.
-- You can use CHATGPT to convert code to idimatic style, to generate documentation.
-->
## Which issue does this PR close?

<!--
We generally require a GitHub issue to be filed for all bug fixes and enhancements and this helps us generate change logs for our releases. You can link an issue to this PR using the GitHub syntax. For example `Closes #123` indicates that this PR will close issue #123.
-->

Closes #.

## Rationale for this change

<!--
 Why are you proposing this change? If this is already explained clearly in the issue then this section is not needed.
 Explaining clearly why changes are proposed helps reviewers understand your changes and offer better suggestions for fixes.  
-->

## What changes are included in this PR?

<!--
There is no need to duplicate the description in the issue here but it is sometimes worth providing a summary of the individual changes in this PR.
-->

## Are these changes tested?

<!--
We typically require tests for all PRs in order to:
1. Prevent the code from being accidentally broken by subsequent changes
2. Serve as another way to document the expected behavior of the code

If tests are not included in your PR, please explain why (for example, are they covered by existing tests)?
-->

## Are there any user-facing changes?

<!--
If there are user-facing changes then we may require documentation to be updated before approving the PR.
-->

<!--
If there are any breaking changes to public APIs, please add the `api change` label.
-->
