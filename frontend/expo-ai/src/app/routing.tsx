import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { StatusCard } from '@/components/status-card';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useRuntimeFeed } from '@/hooks/use-runtime-feed';
import { fallbackRoutes } from '@/lib/runtime';

export default function RoutingScreen() {
  const { data, error } = useRuntimeFeed('/api/runtime/routes', fallbackRoutes);

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content}>
          <ThemedText type="title">Routing map</ThemedText>
          <ThemedText style={styles.subtitle}>
            Expo Router handles the mobile shell while the hosted runtime publishes the API contract the app reads on boot.
          </ThemedText>

          <StatusCard eyebrow="App screens" title="Client routes">
            {data.screens.map((route) => (
              <View key={route.path} style={styles.row}>
                <ThemedText type="smallBold">{route.path}</ThemedText>
                <ThemedText>{route.screen}</ThemedText>
                <ThemedText type="small" themeColor="textSecondary">
                  {route.purpose}
                </ThemedText>
              </View>
            ))}
          </StatusCard>

          <StatusCard eyebrow="API routes" title="Hosted endpoints" accent="amber">
            {data.api_routes.map((route) => (
              <View key={route.path} style={styles.row}>
                <ThemedText type="smallBold">{route.path}</ThemedText>
                <ThemedText type="small" themeColor="textSecondary">
                  {route.purpose}
                </ThemedText>
              </View>
            ))}
          </StatusCard>

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
  row: {
    gap: Spacing.one,
    paddingBottom: Spacing.two,
  },
});
