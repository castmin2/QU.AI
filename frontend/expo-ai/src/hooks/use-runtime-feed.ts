import { startTransition, useEffect, useEffectEvent, useState } from 'react';

import { fetchRuntimeJson, RuntimeRequestState } from '@/lib/runtime';

export function useRuntimeFeed<T>(path: string, fallback: T): RuntimeRequestState<T> {
  const [data, setData] = useState<T>(fallback);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useEffectEvent(async () => {
    setLoading(true);
    const next = await fetchRuntimeJson(path, fallback);
    startTransition(() => {
      setData(next.data);
      setError(next.error);
      setLoading(false);
    });
  });

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, error };
}
