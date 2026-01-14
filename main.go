package main

import (
	"fmt"
	"os"

	"github.com/saga/llm"
)

func main() {
	result, err := llm.Call("What is the capital of France?")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Response: %s\n\n", result.Response)
	fmt.Printf("Tokens (%d): %v\n", len(result.Tokens), result.Tokens)
}
