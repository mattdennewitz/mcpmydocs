package logger

import (
	"io"
	"log/slog"
	"os"
)

var defaultLogger *slog.Logger

func init() {
	// Default to discarding logs (quiet mode)
	defaultLogger = slog.New(slog.NewTextHandler(io.Discard, nil))
}

// Init initializes the logger with the specified verbosity level.
// If verbose is true, logs are written to stderr at Debug level.
// Otherwise, logs are discarded.
func Init(verbose bool) {
	if verbose {
		opts := &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}
		defaultLogger = slog.New(slog.NewTextHandler(os.Stderr, opts))
	} else {
		defaultLogger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	slog.SetDefault(defaultLogger)
}

// Debug logs a debug message with optional key-value pairs.
func Debug(msg string, args ...any) {
	defaultLogger.Debug(msg, args...)
}

// Info logs an info message with optional key-value pairs.
func Info(msg string, args ...any) {
	defaultLogger.Info(msg, args...)
}

// Warn logs a warning message with optional key-value pairs.
func Warn(msg string, args ...any) {
	defaultLogger.Warn(msg, args...)
}

// Error logs an error message with optional key-value pairs.
func Error(msg string, args ...any) {
	defaultLogger.Error(msg, args...)
}
