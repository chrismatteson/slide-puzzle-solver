import { describe, expect, it } from "vitest";
import { type PuzzleProfile } from "../src/picture/profile.ts";
import {
  MAX_PROFILES,
  type ProfileStore,
  STORAGE_KEY,
  forgetProfiles,
  loadProfiles,
  saveProfile,
} from "../src/app/profiles.ts";

/** An in-memory store; `capacity` is how many characters it accepts in total (like a quota). */
function createStore(capacity = Infinity): ProfileStore & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      if (value.length > capacity) throw new DOMException("full", "QuotaExceededError");
      data.set(key, value);
    },
    removeItem: (key) => {
      data.delete(key);
    },
  };
}

function createProfileStub(seed: number): PuzzleProfile {
  const tile = btoa(String.fromCharCode(...new Uint8Array(768).fill(seed)));
  return { version: 1, size: 3, blankHome: 0, tiles: new Array<string>(8).fill(tile) };
}

describe("saved puzzles", () => {
  it("loads nothing from an empty, missing or damaged store", () => {
    expect(loadProfiles(undefined)).toEqual([]);
    expect(loadProfiles(createStore())).toEqual([]);
    const damaged = createStore();
    damaged.setItem(STORAGE_KEY, "{not json");
    expect(loadProfiles(damaged)).toEqual([]);
    damaged.setItem(STORAGE_KEY, JSON.stringify({ not: "a list" }));
    expect(loadProfiles(damaged)).toEqual([]);
  });

  it("drops entries that are not profiles and keeps the rest", () => {
    const store = createStore();
    const good = createProfileStub(1);
    const wrongTileCount = { ...createProfileStub(2), tiles: ["x"] };
    store.setItem(STORAGE_KEY, JSON.stringify([wrongTileCount, 7, null, good]));
    expect(loadProfiles(store)).toEqual([good]);
  });

  it("saves the newest first and keeps at most MAX_PROFILES", () => {
    const store = createStore();
    for (let seed = 1; seed <= MAX_PROFILES + 2; seed++) {
      expect(saveProfile(store, createProfileStub(seed))).toBe(true);
    }
    const saved = loadProfiles(store);
    expect(saved).toHaveLength(MAX_PROFILES);
    expect(saved[0]).toEqual(createProfileStub(MAX_PROFILES + 2));
  });

  it("replaces the puzzle it supersedes instead of adding a copy", () => {
    const store = createStore();
    const first = createProfileStub(1);
    saveProfile(store, first);
    saveProfile(store, createProfileStub(2));
    saveProfile(store, createProfileStub(3), first);
    expect(loadProfiles(store)).toEqual([createProfileStub(3), createProfileStub(2)]);
  });

  it("drops older puzzles when the store is full, and reports failure when nothing fits", () => {
    const one = JSON.stringify([createProfileStub(1)]).length;
    const store = createStore(one + 10);
    saveProfile(store, createProfileStub(1));
    expect(saveProfile(store, createProfileStub(2))).toBe(true);
    expect(loadProfiles(store)).toEqual([createProfileStub(2)]);
    expect(saveProfile(createStore(10), createProfileStub(3))).toBe(false);
    expect(saveProfile(undefined, createProfileStub(3))).toBe(false);
  });

  it("forgets everything, even from a store that throws", () => {
    const store = createStore();
    saveProfile(store, createProfileStub(1));
    forgetProfiles(store);
    expect(loadProfiles(store)).toEqual([]);
    expect(() => {
      forgetProfiles({
        ...store,
        removeItem: () => {
          throw new Error("denied");
        },
      });
    }).not.toThrow();
  });
});
