import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { StatusCard } from '@/components/status-card';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useRuntimeFeed } from '@/hooks/use-runtime-feed';
import { fallbackUpdates } from '@/lib/runtime';

export default function UpdatesScreen() {
  const { data, error } = useRuntimeFeed('/api/runtime/updates', fallbackUpdates);

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <ThemedText type="title">Runtime updates</ThemedText>
          <ThemedText style={styles.subtitle}>
            This feed is where the hosted app keeps deployment and pack changes visible without shipping a new client.
          </ThemedText>
          {data.items.map((item, index) => (
            <StatusCard key={item.id} eyebrow={`Update ${index + 1}`} title={item.title} accent={index % 2 ? 'amber' : 'mint'}>
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
