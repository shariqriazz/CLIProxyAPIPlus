package test

import (
	"context"
	"strings"
	"testing"

	geminicli_claude "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini-cli/claude"
	gemini_claude "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/claude"
)

// TestGeminiCLIClaudeSignatureDelta tests that signature_delta is properly emitted
// when a thoughtSignature is present in the Gemini CLI response.
func TestGeminiCLIClaudeSignatureDelta(t *testing.T) {
	// Request JSON with user content for session ID derivation
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	// Response with thinking and signature
	responseWithSignature := []byte(`{
		"response": {
			"modelVersion": "gemini-3-pro-preview",
			"responseId": "test-response-id",
			"candidates": [{
				"content": {
					"parts": [{
						"text": "Thinking about the answer...",
						"thought": true,
						"thoughtSignature": "ABC123XYZ_signature_data"
					}]
				}
			}]
		}
	}`)

	var param any = nil
	results := geminicli_claude.ConvertGeminiCLIResponseToClaude(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseWithSignature, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	output := strings.Join(results, "")

	// Should contain signature_delta event
	if !strings.Contains(output, "signature_delta") {
		t.Errorf("Expected signature_delta in output, got: %s", output)
	}

	// Should contain the actual signature value
	if !strings.Contains(output, "ABC123XYZ_signature_data") {
		t.Errorf("Expected signature value in output, got: %s", output)
	}

	// Should also have thinking_delta for the text
	if !strings.Contains(output, "thinking_delta") {
		t.Errorf("Expected thinking_delta in output for thinking text, got: %s", output)
	}
}

// TestGeminiCLIClaudeNoSignatureWhenMissing tests that signature_delta is NOT emitted
// when there is no thoughtSignature in the response (just regular thinking).
func TestGeminiCLIClaudeNoSignatureWhenMissing(t *testing.T) {
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	// Response with thinking but NO signature
	responseWithoutSignature := []byte(`{
		"response": {
			"modelVersion": "gemini-3-pro-preview",
			"responseId": "test-response-id",
			"candidates": [{
				"content": {
					"parts": [{
						"text": "Just thinking without signature",
						"thought": true
					}]
				}
			}]
		}
	}`)

	var param any = nil
	results := geminicli_claude.ConvertGeminiCLIResponseToClaude(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseWithoutSignature, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	output := strings.Join(results, "")

	// Should NOT contain signature_delta event
	if strings.Contains(output, "signature_delta") {
		t.Errorf("Should not have signature_delta when no signature present, got: %s", output)
	}

	// Should still have thinking_delta
	if !strings.Contains(output, "thinking_delta") {
		t.Errorf("Expected thinking_delta in output, got: %s", output)
	}
}

// TestGeminiClaudeSignatureDelta tests that signature_delta is properly emitted
// for standard Gemini API format.
func TestGeminiClaudeSignatureDelta(t *testing.T) {
	requestJSON := []byte(`{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}`)

	// Response with thinking and signature (standard Gemini format - no "response" wrapper)
	responseWithSignature := []byte(`{
		"modelVersion": "gemini-3-pro-preview",
		"responseId": "test-response-id",
		"candidates": [{
			"content": {
				"parts": [{
					"text": "Deep thinking here...",
					"thought": true,
					"thoughtSignature": "SIGNATURE_DATA_HERE"
				}]
			}
		}]
	}`)

	var param any = nil
	results := gemini_claude.ConvertGeminiResponseToClaude(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseWithSignature, &param)

	if len(results) == 0 {
		t.Fatal("Expected at least one result")
	}

	output := strings.Join(results, "")

	// Should contain signature_delta event
	if !strings.Contains(output, "signature_delta") {
		t.Errorf("Expected signature_delta in output, got: %s", output)
	}

	// Should contain the actual signature value
	if !strings.Contains(output, "SIGNATURE_DATA_HERE") {
		t.Errorf("Expected signature value in output, got: %s", output)
	}
}

// TestGeminiCLINonStreamSignature tests that signatures are properly included
// in non-streaming responses.
func TestGeminiCLINonStreamSignature(t *testing.T) {
	requestJSON := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}}`)

	responseWithSignature := []byte(`{
		"response": {
			"modelVersion": "gemini-3-pro-preview",
			"responseId": "test-response-id",
			"usageMetadata": {
				"promptTokenCount": 10,
				"candidatesTokenCount": 20,
				"thoughtsTokenCount": 50,
				"totalTokenCount": 80
			},
			"candidates": [{
				"content": {
					"parts": [
						{
							"text": "Thinking deeply...",
							"thought": true,
							"thoughtSignature": "NON_STREAM_SIG_123"
						},
						{
							"text": "Final answer"
						}
					]
				},
				"finishReason": "STOP"
			}]
		}
	}`)

	result := geminicli_claude.ConvertGeminiCLIResponseToClaudeNonStream(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseWithSignature, nil)

	// Should contain signature in the thinking block
	if !strings.Contains(result, `"signature"`) {
		t.Errorf("Expected signature field in non-streaming response, got: %s", result)
	}

	if !strings.Contains(result, "NON_STREAM_SIG_123") {
		t.Errorf("Expected signature value in non-streaming response, got: %s", result)
	}
}

// TestGeminiNonStreamSignature tests non-streaming signature handling for standard Gemini API.
func TestGeminiNonStreamSignature(t *testing.T) {
	requestJSON := []byte(`{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}`)

	responseWithSignature := []byte(`{
		"modelVersion": "gemini-3-pro-preview",
		"responseId": "test-response-id",
		"usageMetadata": {
			"promptTokenCount": 10,
			"candidatesTokenCount": 20,
			"thoughtsTokenCount": 50,
			"totalTokenCount": 80
		},
		"candidates": [{
			"content": {
				"parts": [
					{
						"text": "Reasoning here...",
						"thought": true,
						"thoughtSignature": "GEMINI_SIG_456"
					},
					{
						"text": "The answer is 42"
					}
				]
			},
			"finishReason": "STOP"
		}]
	}`)

	result := gemini_claude.ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-pro-preview", requestJSON, requestJSON, responseWithSignature, nil)

	// Should contain signature in the thinking block
	if !strings.Contains(result, `"signature"`) {
		t.Errorf("Expected signature field in non-streaming response, got: %s", result)
	}

	if !strings.Contains(result, "GEMINI_SIG_456") {
		t.Errorf("Expected signature value in non-streaming response, got: %s", result)
	}
}
