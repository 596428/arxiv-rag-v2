// OpenAI Embedding API for query embedding
import type { EmbeddingResult } from './types.ts';

const OPENAI_API_URL = 'https://api.openai.com/v1/embeddings';

export async function embedQuery(
  query: string,
  apiKey: string
): Promise<EmbeddingResult> {
  const startTime = performance.now();

  const response = await fetch(OPENAI_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'text-embedding-3-large',
      input: query,
      dimensions: 3072,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  const embedding = data.data[0].embedding;
  const time_ms = Math.round(performance.now() - startTime);

  return { embedding, time_ms };
}
