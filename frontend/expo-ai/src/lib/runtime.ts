export type FeedItem = {
  id: string;
  title: string;
  detail: string;
};

export type ScreenRoute = {
  path: string;
  screen: string;
  purpose: string;
};

export type ApiRoute = {
  path: string;
  purpose: string;
};

export type RuntimeBootstrap = {
  project: string;
  default_bundle: string;
  bundle: {
    model_name: string;
    pack_path: string;
    state_dict_hash_sha256: string;
    pack_download?: {
      provider?: string;
      direct_url?: string;
      size_bytes?: number;
      pack_file_sha256?: string;
    };
  };
  pack_status: {
    exists: boolean;
    actual_size: number;
    expected_size: number;
    size_matches: boolean;
  };
  oracle_override_active: boolean;
  prefetch?: {
    error?: string;
    path?: string;
  };
};

export type RouteFeed = {
  screens: ScreenRoute[];
  api_routes: ApiRoute[];
};

export type RuntimeRequestState<T> = {
  data: T;
  loading: boolean;
  error: string;
};

export const runtimeBaseUrl = (process.env.EXPO_PUBLIC_QUAI_RUNTIME_URL ?? 'http://localhost:8787').replace(/\/+$/, '');

export const fallbackBootstrap: RuntimeBootstrap = {
  project: 'QU.AI',
  default_bundle: 'qwen2.5-coder-3b',
  bundle: {
    model_name: 'qwen2.5-coder:3b',
    pack_path: 'ai/runtime/chatbox/packs/qwen2.5-coder_3b.rftmwpk',
    state_dict_hash_sha256: '7ad9899be6aa2643efb160d044253ca7866b0e3525c986a2428c2a4ce72784c3',
    pack_download: {
      provider: 'oracle_object_storage',
      size_bytes: 2128004908,
      pack_file_sha256: '502112d74e76d8d1d38c6189a038cc45ffd229fb773f557067dc21c3355e0d33',
    },
  },
  pack_status: {
    exists: false,
    actual_size: 0,
    expected_size: 2128004908,
    size_matches: false,
  },
  oracle_override_active: false,
  prefetch: {
    error: '',
    path: '',
  },
};

export const fallbackUpdates: { items: FeedItem[] } = {
  items: [
    {
      id: 'pack-default',
      title: 'qwen direct-GGUF pack is the active default',
      detail: 'The mobile shell assumes qwen2.5-coder:3b and lets the hosted runtime decide where the pack binary comes from.',
    },
    {
      id: 'oracle-feed',
      title: 'Oracle bucket override is first-class',
      detail: 'Hosted deployments can point the prerequisite download at an Oracle pre-authenticated object URL without changing the committed manifest.',
    },
  ],
};

export const fallbackResearch: { items: FeedItem[] } = {
  items: [
    {
      id: 'expo-router',
      title: 'Expo Router anchors the app shell',
      detail: 'The client uses file-based routes so runtime, updates, research, and routing screens can evolve without a custom navigation registry.',
    },
    {
      id: 'narrow-contract',
      title: 'Keep the hosted contract narrow',
      detail: 'The backend only needs bootstrap, updates, research, and route metadata while heavier model work stays behind the runtime boundary.',
    },
  ],
};

export const fallbackRoutes: RouteFeed = {
  screens: [
    { path: '/', screen: 'Runtime', purpose: 'Show bundle and host readiness.' },
    { path: '/updates', screen: 'Updates', purpose: 'Show runtime deltas and deployment notes.' },
    { path: '/research', screen: 'Research', purpose: 'Show hosting and storage notes.' },
    { path: '/routing', screen: 'Routing', purpose: 'Show screen and API route maps.' },
  ],
  api_routes: [
    { path: '/healthz', purpose: 'Hosted health and pack readiness.' },
    { path: '/api/runtime/bootstrap', purpose: 'Bundle contract for the client.' },
  ],
};

export async function fetchRuntimeJson<T>(path: string, fallback: T): Promise<{ data: T; error: string }> {
  try {
    const response = await fetch(`${runtimeBaseUrl}${path}`);
    if (!response.ok) {
      return { data: fallback, error: `runtime returned ${response.status}` };
    }
    const data = (await response.json()) as T;
    return { data, error: '' };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'runtime unavailable';
    return { data: fallback, error: message };
  }
}
