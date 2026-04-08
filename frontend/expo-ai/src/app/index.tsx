import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { StatusCard } from '@/components/status-card';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useRuntimeFeed } from '@/hooks/use-runtime-feed';
import { fallbackBootstrap, runtimeBaseUrl } from '@/lib/runtime';

export default function RuntimeScreen() {
  const { data, loading, error } = useRuntimeFeed('/api/runtime/bootstrap', fallbackBootstrap);
  const provider = data.bundle.pack_download?.provider || 'external_http';
  const expectedSizeGb = ((data.pack_status.expected_size || 0) / 1_000_000_000).toFixed(2);

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <View style={styles.hero}>
            <ThemedText type="smallBold" themeColor="textSecondary" style={styles.kicker}>
              HOSTED QU.AI RUNTIME
            </ThemedText>
            <ThemedText type="title" style={styles.title}>
              Pack-aware mobile shell for your qwen weights.
            </ThemedText>
            <ThemedText style={styles.subtitle}>
              This screen reads the hosted runtime contract first, then lets the backend decide whether the prerequisite pack comes from Oracle Object Storage or the fallback external host.
            </ThemedText>
          </View>

          <View style={styles.grid}>
            <StatusCard eyebrow="Bundle" title={data.bundle.model_name || data.default_bundle}>
              <ThemedText>Default bundle: {data.default_bundle}</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Pack path: {data.bundle.pack_path}
              </ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                State hash: {data.bundle.state_dict_hash_sha256}
              </ThemedText>
            </StatusCard>

            <StatusCard eyebrow="Prerequisite" title={`${expectedSizeGb} GB pack`} accent="amber">
              <ThemedText>Provider: {provider}</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Oracle override active: {data.oracle_override_active ? 'yes' : 'no'}
              </ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Pack ready: {data.pack_status.exists && data.pack_status.size_matches ? 'yes' : 'not yet'}
              </ThemedText>
            </StatusCard>

            <StatusCard eyebrow="Host" title={runtimeBaseUrl}>
              <ThemedText>Runtime endpoint is read from `EXPO_PUBLIC_QUAI_RUNTIME_URL`.</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Loading: {loading ? 'yes' : 'no'}
              </ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Error: {error || data.prefetch?.error || 'none'}
              </ThemedText>
            </StatusCard>
          </View>
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  content: {
    alignSelf: 'center',
    width: '100%',
    maxWidth: MaxContentWidth,
    paddingHorizontal: Spacing.four,
    paddingTop: Spacing.six,
    paddingBottom: BottomTabInset + Spacing.six,
    gap: Spacing.five,
  },
  hero: {
    gap: Spacing.three,
  },
  kicker: {
    letterSpacing: 1.5,
  },
  title: {
    maxWidth: 720,
  },
  subtitle: {
    maxWidth: 720,
    fontSize: 18,
    lineHeight: 28,
  },
  grid: {
    gap: Spacing.three,
  },
});
