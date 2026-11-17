# White-box split test follows the explicit factory init contract

`ExplicitDataFactory` resolves the declared split in `_init_data` before
`_init_dataset` consumes `self.split_result`. The white-box construction in
`test_grouped_splitting_core.py` bypassed `_init_data`, so after the split
resolution moved there the test failed with a missing `split_result`
attribute. The test now mirrors the public init contract explicitly instead
of relying on the old internal ordering. No production behavior changes.
