package llm

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"unicode"

	"github.com/saga/checkpointing"
)

func Call(prompt string) (*Result, error) {
	// Generate unique request ID
	reqID := generateUUID()

	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("HUGGINGFACE_API_KEY environment variable not set")
	}

	apiURL := "https://router.huggingface.co/v1/chat/completions"

	// Limit to 10 tokens max
	reqBody := ChatRequest{
		Model: "Qwen/Qwen2.5-Coder-32B-Instruct",
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens: 10,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	responseText := chatResp.Choices[0].Message.Content
	tokens := tokenize(responseText)

	// Save checkpoint with raw text
	checkpoint := checkpointing.Checkpoint{
		ReqID:         reqID,
		TokensSoFar:   responseText,
		InitialPrompt: prompt,
	}
	if err := checkpointing.Save(checkpoint); err != nil {
		return nil, fmt.Errorf("failed to save checkpoint: %w", err)
	}

	return &Result{
		ReqID:    reqID,
		Response: responseText,
		Tokens:   tokens,
	}, nil
}

func tokenize(text string) []string {
	var tokens []string
	var currentToken strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
			if !unicode.IsSpace(r) {
				tokens = append(tokens, string(r))
			}
		} else {
			currentToken.WriteRune(r)
		}
	}

	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	return tokens
}

func Resume(reqID string) (*Result, error) {
	// Load checkpoint
	checkpoint, err := checkpointing.LoadByID(reqID)
	if err != nil {
		return nil, fmt.Errorf("failed to load checkpoint: %w", err)
	}

	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("HUGGINGFACE_API_KEY environment variable not set")
	}

	apiURL := "https://router.huggingface.co/v1/chat/completions"

	// Build conversation history with explicit continuation instruction
	reqBody := ChatRequest{
		Model: "Qwen/Qwen2.5-Coder-32B-Instruct",
		Messages: []Message{
			{
				Role:    "user",
				Content: checkpoint.InitialPrompt,
			},
			{
				Role:    "assistant",
				Content: checkpoint.TokensSoFar,
			},
			{
				Role:    "user",
				Content: "Continue from where you left off. Do not repeat any text. Start with the very next token/word.",
			},
		},
		MaxTokens: 10,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	responseText := chatResp.Choices[0].Message.Content

	// Combine previous and new text
	fullResponse := checkpoint.TokensSoFar + " " + responseText
	tokens := tokenize(fullResponse)

	// Update checkpoint
	updatedCheckpoint := checkpointing.Checkpoint{
		ReqID:         reqID,
		TokensSoFar:   fullResponse,
		InitialPrompt: checkpoint.InitialPrompt,
	}
	if err := checkpointing.Save(updatedCheckpoint); err != nil {
		return nil, fmt.Errorf("failed to update checkpoint: %w", err)
	}

	return &Result{
		ReqID:    reqID,
		Response: fullResponse,
		Tokens:   tokens,
	}, nil
}

func generateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
