package main

import (
	"fmt"
	"os"

	"github.com/saga/checkpointing"
	"github.com/saga/llm"
)

func main() {
	prompt := "Write a short paragraph explaining how photosynthesis works in plants."

	// Initial call - generates 10 tokens and saves checkpoint
	fmt.Println("=== Initial Call ===")
	fmt.Printf("Prompt: %s\n\n", prompt)

	result, err := llm.Call(prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Request ID: %s\n", result.ReqID)
	fmt.Printf("Response (10 tokens max): %s\n", result.Response)
	fmt.Printf("Tokens (%d): %v\n\n", len(result.Tokens), result.Tokens)

	// Show what's in the checkpoint
	fmt.Println("=== Checkpoint Saved ===")
	checkpoint, err := checkpointing.LoadByID(result.ReqID)
	if err != nil {
		fmt.Printf("Error loading checkpoint: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Initial Prompt: %s\n", checkpoint.InitialPrompt)
	fmt.Printf("Tokens So Far: %s\n\n", checkpoint.TokensSoFar)

	// Keep resuming until we have around 50 tokens
	fmt.Println("=== Resuming until ~50 tokens ===")
	currentResult := result
	targetTokens := 50
	iteration := 1

	for len(currentResult.Tokens) < targetTokens {
		fmt.Printf("\n[Iteration %d] Resuming (current: %d tokens, target: %d)...\n", iteration, len(currentResult.Tokens), targetTokens)

		resumedResult, err := llm.Resume(currentResult.ReqID)
		if err != nil {
			fmt.Printf("Error resuming: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("New tokens generated: %d\n", len(resumedResult.Tokens)-len(currentResult.Tokens))
		fmt.Printf("Total tokens now: %d\n", len(resumedResult.Tokens))
		fmt.Printf("Current text: %s\n", resumedResult.Response)

		currentResult = resumedResult
		iteration++
	}

	// Show final result
	fmt.Println("\n=== Final Result ===")
	fmt.Printf("Request ID: %s\n", currentResult.ReqID)
	fmt.Printf("Full Response: %s\n", currentResult.Response)
	fmt.Printf("Total Tokens: %d\n\n", len(currentResult.Tokens))

	// Show final checkpoint
	fmt.Println("=== Final Checkpoint ===")
	finalCheckpoint, err := checkpointing.LoadByID(currentResult.ReqID)
	if err != nil {
		fmt.Printf("Error loading checkpoint: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Initial Prompt: %s\n", finalCheckpoint.InitialPrompt)
	fmt.Printf("Tokens So Far: %s\n", finalCheckpoint.TokensSoFar)
}
