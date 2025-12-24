package test

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"

	antigravity_openai "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/antigravity/openai/chat-completions"
	geminicli_openai "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini-cli/openai/chat-completions"
)

// TestAntigravityOpenAITokenAccounting verifies that thoughtsTokenCount goes to completion_tokens
// (output tokens) not prompt_tokens (input tokens).
func TestAntigravityOpenAITokenAccounting(t *testing.T) {
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	// Response with thinking tokens
	responseJSON := []byte(`{
		"response": {
			"modelVersion": "gemini-3-pro-preview",
			"responseId": "test-response-id",
			"usageMetadata": {
				"promptTokenCount": 100,
				"candidatesTokenCount": 50,
				"thoughtsTokenCount": 200,
				"totalTokenCount": 350
			},
			"candidates": [{
				"content": {
					"parts": [{
						"text": "The answer is 42"
					}]
				},
				"finishReason": "STOP"
			}]
		}
	}`)

	var param any = nil
	results := antigravity_openai.ConvertAntigravityResponseToOpenAI(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseJSON, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	result := results[0]

	// Parse the result
	parsed := gjson.Parse(result)

	promptTokens := parsed.Get("usage.prompt_tokens").Int()
	completionTokens := parsed.Get("usage.completion_tokens").Int()
	reasoningTokens := parsed.Get("usage.completion_tokens_details.reasoning_tokens").Int()

	// Prompt tokens should be ONLY promptTokenCount (100), NOT including thoughtsTokenCount
	if promptTokens != 100 {
		t.Errorf("prompt_tokens should be 100 (promptTokenCount only), got: %d", promptTokens)
	}

	// Completion tokens should include candidatesTokenCount + thoughtsTokenCount = 50 + 200 = 250
	expectedCompletion := int64(250)
	if completionTokens != expectedCompletion {
		t.Errorf("completion_tokens should be %d (candidates + thoughts), got: %d", expectedCompletion, completionTokens)
	}

	// Reasoning tokens should be thoughtsTokenCount (200)
	if reasoningTokens != 200 {
		t.Errorf("reasoning_tokens should be 200, got: %d", reasoningTokens)
	}
}

// TestGeminiCLIOpenAITokenAccounting verifies token accounting for Gemini CLI to OpenAI translation.
func TestGeminiCLIOpenAITokenAccounting(t *testing.T) {
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	// Response with thinking tokens
	responseJSON := []byte(`{
		"response": {
			"modelVersion": "gemini-3-flash-preview",
			"responseId": "test-response-id",
			"usageMetadata": {
				"promptTokenCount": 80,
				"candidatesTokenCount": 30,
				"thoughtsTokenCount": 150,
				"totalTokenCount": 260
			},
			"candidates": [{
				"content": {
					"parts": [{
						"text": "Response text"
					}]
				},
				"finishReason": "STOP"
			}]
		}
	}`)

	var param any = nil
	results := geminicli_openai.ConvertCliResponseToOpenAI(context.Background(), "gemini-3-flash-preview", requestJSON, requestJSON, responseJSON, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	result := results[0]
	parsed := gjson.Parse(result)

	promptTokens := parsed.Get("usage.prompt_tokens").Int()
	completionTokens := parsed.Get("usage.completion_tokens").Int()
	reasoningTokens := parsed.Get("usage.completion_tokens_details.reasoning_tokens").Int()

	// Prompt tokens should be ONLY promptTokenCount (80)
	if promptTokens != 80 {
		t.Errorf("prompt_tokens should be 80 (promptTokenCount only), got: %d", promptTokens)
	}

	// Completion tokens should include candidatesTokenCount + thoughtsTokenCount = 30 + 150 = 180
	expectedCompletion := int64(180)
	if completionTokens != expectedCompletion {
		t.Errorf("completion_tokens should be %d (candidates + thoughts), got: %d", expectedCompletion, completionTokens)
	}

	// Reasoning tokens should be thoughtsTokenCount (150)
	if reasoningTokens != 150 {
		t.Errorf("reasoning_tokens should be 150, got: %d", reasoningTokens)
	}
}

// TestTokenAccountingWithZeroThinkingTokens verifies behavior when there are no thinking tokens.
func TestTokenAccountingWithZeroThinkingTokens(t *testing.T) {
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	// Response without thinking tokens
	responseJSON := []byte(`{
		"response": {
			"modelVersion": "gemini-2.5-pro",
			"responseId": "test-response-id",
			"usageMetadata": {
				"promptTokenCount": 50,
				"candidatesTokenCount": 100,
				"totalTokenCount": 150
			},
			"candidates": [{
				"content": {
					"parts": [{
						"text": "No thinking here"
					}]
				},
				"finishReason": "STOP"
			}]
		}
	}`)

	var param any = nil
	results := antigravity_openai.ConvertAntigravityResponseToOpenAI(context.Background(), "gemini-2.5-pro", requestJSON, requestJSON, responseJSON, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	result := results[0]
	parsed := gjson.Parse(result)

	promptTokens := parsed.Get("usage.prompt_tokens").Int()
	completionTokens := parsed.Get("usage.completion_tokens").Int()
	reasoningTokens := parsed.Get("usage.completion_tokens_details.reasoning_tokens")

	// Prompt tokens should be 50
	if promptTokens != 50 {
		t.Errorf("prompt_tokens should be 50, got: %d", promptTokens)
	}

	// Completion tokens should be 100 (candidatesTokenCount only)
	if completionTokens != 100 {
		t.Errorf("completion_tokens should be 100, got: %d", completionTokens)
	}

	// reasoning_tokens should not be present when thoughtsTokenCount is 0
	if reasoningTokens.Exists() && reasoningTokens.Int() != 0 {
		t.Errorf("reasoning_tokens should not exist or be 0 when no thinking, got: %d", reasoningTokens.Int())
	}
}
