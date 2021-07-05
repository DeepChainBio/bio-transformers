# Logging

When running bio-transformers, printed messages are entirely controles by bio-transformers code.
The log level controls which types of log messages would be printed.

bio-transformers uses the Python module ``logging` <https://docs.python.org/3/library/logging.html>`_ to log the messages. The log level is controlled by the environment variable BIO_LOG_LEVEL. The levels are given in the table below. The default level is “2”.

To adjust the logging level, you can export the environment variable:

  ```bash
  export BIO_LOG_LEVEL=1
  ```

  | DEEPREG_LOG_LEVEL | Behavior                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------ |
| "0"               | Log all messages, equivalent to `logging.DEBUG`. Same as log level "1".                    |
| "1"               | Log all messages, equivalent to `logging.DEBUG`.                                           |
| "2"               | Log all messages except DEBUG, equivalent to `logging.INFO`. (default)                     |
| "3"               | Log all messages except DEBUG and INFO, equivalent to `logging.WARNING`.                   |
| "4"               | Log all messages except DEBUG, INFO, and WARNING, equivalent to `logging.ERROR`.           |
| "5"               | Log all messages except DEBUG, INFO, WARNING, and ERROR, equivalent to `logging.CRITICAL`. |
