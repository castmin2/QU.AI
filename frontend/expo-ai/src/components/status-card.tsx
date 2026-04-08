import React, { PropsWithChildren } from 'react';
import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Spacing } from '@/constants/theme';

type StatusCardProps = PropsWithChildren<{
  eyebrow?: string;
  title: string;
  accent?: 'mint' | 'amber';
}>;

export function StatusCard({ accent = 'mint', eyebrow, title, children }: StatusCardProps) {
  return (
    <ThemedView type="backgroundElement" style={styles.card}>
      <View style={[styles.accentBar, accent === 'amber' ? styles.amber : styles.mint]} />
      {!!eyebrow && (
        <ThemedText type="smallBold" themeColor="textSecondary" style={styles.eyebrow}>
          {eyebrow}
        </ThemedText>
      )}
      <ThemedText type="subtitle" style={styles.title}>
        {title}
      </ThemedText>
      <View style={styles.content}>{children}</View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 28,
    padding: Spacing.four,
    gap: Spacing.two,
  },
  accentBar: {
    width: 72,
    height: 6,
    borderRadius: 999,
  },
  mint: {
    backgroundColor: '#2E8B7D',
  },
  amber: {
    backgroundColor: '#C46B2A',
  },
  eyebrow: {
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  title: {
    fontSize: 24,
    lineHeight: 30,
  },
  content: {
    gap: Spacing.two,
  },
});
