import { Message } from '@/types/chat';
import { OpenAIModel } from '@/types/openai';

import { AZURE_DEPLOYMENT_ID, OPENAI_API_HOST, OPENAI_API_TYPE, OPENAI_API_VERSION, OPENAI_ORGANIZATION } from '../app/const';

import {
  ParsedEvent,
  ReconnectInterval,
  createParser,
} from 'eventsource-parser';

export class OpenAIError extends Error {
  type: string;
  param: string;
  code: string;

  constructor(message: string, type: string, param: string, code: string) {
    super(message);
    this.name = 'OpenAIError';
    this.type = type;
    this.param = param;
    this.code = code;
  }
}

export const OpenAIStream = async (
  model: OpenAIModel,
  systemPrompt: string,
  temperature: number,
  key: string,
  messages: Message[],
) => {
  let url = `${process.env.API_URL}/tweets_olivia/finetune`;
  if (OPENAI_API_TYPE === 'azure') {
    url = `${OPENAI_API_HOST}/openai/deployments/${AZURE_DEPLOYMENT_ID}/chat/completions?api-version=${OPENAI_API_VERSION}`;
  }

  // Validate and sanitize messages
  const sanitizedMessages = messages.map((msg) => ({
    role: msg.role,
    content: msg.content?.replace(/[\r\n\t]/g, '').trim(), // Remove problematic characters
  }));

  const requestBody = {
    ...(OPENAI_API_TYPE === 'openai' && { model: model.id }),
    messages: [
      {
        role: 'system',
        content: systemPrompt,
      },
      ...sanitizedMessages,
    ],
    max_tokens: 5000,
    temperature,
    stream: false,
  };

  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(OPENAI_API_TYPE === 'openai' && {
          Authorization: `Bearer ${key || process.env.OPENAI_API_KEY}`,
        }),
        ...(OPENAI_API_TYPE === 'azure' && {
          'api-key': `${key || process.env.OPENAI_API_KEY}`,
        }),
        ...(OPENAI_API_TYPE === 'openai' && OPENAI_ORGANIZATION && {
          'OpenAI-Organization': OPENAI_ORGANIZATION,
        }),
      },
      method: 'POST',
      body: JSON.stringify(requestBody),
    });

    if (!res.ok) {
      const errorResponse = await res.text();
      throw new Error(
        `Failed to fetch: ${res.status} ${res.statusText}\n${errorResponse}`
      );
    }

    const result = await res.json();
    return result.response;

  } catch (error) {
    console.error('Error in OpenAIStream:', error.message);
    throw error;
  }
};