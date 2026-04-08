import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { StatusCard } from '@/components/status-card';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useRuntimeFeed } from '@/hooks/use-runtime-feed';
import { fallbackResearch } from '@/lib/runtime';

export default function ResearchScreen() {
  const { data, error } = useRuntimeFeed('/api/runtime/research', fallbackResearch);

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <ThemedText type="title">Research notes</ThemedText>
          <ThemedText style={styles.subtitle}>
            The client mirrors the current hosting assumptions so the runtime, storage path, and route strategy stay visible while we build.
          </ThemedText>
          {data.items.map((item) => (
            <StatusCard key={item.id} eyebrow="Research" title={item.title} accent="amber">
              <ThemedText>{item.detail}</ThemedText>
            </StatusCard>
          ))}
          {!!error && (
            <ThemedText type="small" themeColor="textSecondary">
              Runtime fallback in use: {error}
            </ThemedText>
          )}
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
    gap: Spacing.three,
  },
  subtitle: {
    fontSize: 18,
    lineHeight: 28,
    maxWidth: 760,
  },
});
