# Hardware runner examples

Copy `runner.example.json` and replace the commands with the flashing and
capture tool used by your board. The commands must be argv arrays and must
contain `{exe}` where the linked ELF should be passed.

The board firmware must print the same five `index prediction` lines as the
generated harness. Optional measurements use the `KILN_METRIC key=value`
protocol documented in [TARGET_VALIDATION.md](../../TARGET_VALIDATION.md).
