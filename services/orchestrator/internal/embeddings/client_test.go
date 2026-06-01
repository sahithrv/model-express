package embeddings

import (
	"context"
	"reflect"
	"testing"
)

func TestFakeClientReturnsStableVectors(t *testing.T) {
	client := NewFakeClient(6)

	first, err := client.Embed(context.Background(), EmbedRequest{
		Model:      "fake-model",
		Text:       "class balancing improved minority recall",
		Dimensions: 6,
	})
	if err != nil {
		t.Fatalf("Embed() first error = %v", err)
	}
	second, err := client.Embed(context.Background(), EmbedRequest{
		Model:      "fake-model",
		Text:       "class balancing improved minority recall",
		Dimensions: 6,
	})
	if err != nil {
		t.Fatalf("Embed() second error = %v", err)
	}
	other, err := client.Embed(context.Background(), EmbedRequest{
		Model:      "fake-model",
		Text:       "plateaued after early epochs",
		Dimensions: 6,
	})
	if err != nil {
		t.Fatalf("Embed() other error = %v", err)
	}

	if first.Model != "fake-model" || first.Dimensions != 6 || len(first.Vector) != 6 {
		t.Fatalf("unexpected first embedding metadata: %#v", first)
	}
	if !reflect.DeepEqual(first.Vector, second.Vector) {
		t.Fatalf("fake vectors should be stable for the same input")
	}
	if reflect.DeepEqual(first.Vector, other.Vector) {
		t.Fatalf("fake vectors should differ for different input text")
	}
}

func TestDisabledClientErrorsClearly(t *testing.T) {
	client := NewClient(Config{})
	_, err := client.Embed(context.Background(), EmbedRequest{
		Model:      "fake-model",
		Text:       "hello",
		Dimensions: 3,
	})
	if err == nil || err.Error() != ErrDisabled.Error() {
		t.Fatalf("Embed() error = %v, want %v", err, ErrDisabled)
	}
}
