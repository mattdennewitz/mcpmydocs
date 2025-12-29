package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"mcpmydocs/cmd"
	"mcpmydocs/internal/logger"
)

var (
	// Version is set during build via -ldflags
	Version = "0.1.0"

	// verbose enables debug logging
	verbose bool
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "mcpmydocs",
		Short: "mcpmydocs: Local-first Markdown indexing and MCP server",
		Long: `A high-performance CLI tool to chunk, vectorize, and search local
markdown files using ONNX models and DuckDB. Includes a built-in MCP server.`,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			logger.Init(verbose)
		},
	}

	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose debug logging")
	rootCmd.PersistentFlags().StringVar(&cmd.OnnxLibraryPath, "onnx-lib", "", "Path to ONNX Runtime shared library")

	versionCmd := &cobra.Command{
		Use:   "version",
		Short: "Print the version number",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("mcpmydocs %s\n", Version)
		},
	}

	rootCmd.AddCommand(cmd.NewIndexCmd(), cmd.NewSearchCmd(), cmd.NewRunCmd(), versionCmd)

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
