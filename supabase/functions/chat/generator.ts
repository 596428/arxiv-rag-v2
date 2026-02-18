// Gemini API for RAG answer generation
import type { Source, ChatMessage } from './types.ts';

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

export async function generateAnswer(
  query: string,
  sources: Source[],
  history: ChatMessage[],
  apiKey: string
): Promise<{ answer: string; time_ms: number }> {
  const startTime = performance.now();

  // Build context from retrieved chunks
  const context = sources
    .map((s, i) => `[${i + 1}] ${s.title} - ${s.section}\n${s.chunk_text}`)
    .join('\n\n---\n\n');

  // Build conversation history
  const historyText = history
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
    .join('\n');

  const systemPrompt = `You are an expert AI research assistant. Answer questions about machine learning, deep learning, and AI research papers based on the provided context.

Guidelines:
- Base your answers primarily on the provided paper excerpts
- Cite sources using [1], [2], etc. when referencing specific papers
- If the context doesn't contain relevant information, say so clearly
- Be concise but thorough
- Use technical language appropriate for researchers`;

  const userPrompt = `Context from research papers:
${context}

${historyText ? `Previous conversation:\n${historyText}\n\n` : ''}User question: ${query}

Please provide a comprehensive answer based on the context above.`;

  const response = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            { text: systemPrompt },
            { text: userPrompt },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2048,
        topP: 0.95,
      },
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  const answer = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response generated';
  const time_ms = Math.round(performance.now() - startTime);

  return { answer, time_ms };
}
