<template>
        <section class="search-section">
      <input
        v-model="searchQuery"
        @input="onSearchInput"
        placeholder="Search for a game..."
        class="search-input"
      />
      <ul v-if="searchResults.length && searchQuery" class="suggestions-list">
        <li
          v-for="game in searchResults"
          :key="game.id"
          @click="selectGame(game)"
          class="suggestion-item"
        >
          <img :src="game.cover_url" alt="" class="suggestion-image" />
          <span>
            {{ game.release_date ? `${game.name} (${getYear(game.release_date)})` : game.name }}
          </span>

        </li>
      </ul>
    </section>

    <!-- Selected game slots -->
    <section class="selected-games">
      <div
        v-for="(game, index) in selectedGames"
        :key="index"
        class="game-slot"
      >
        <div v-if="game">
          <img :src="game.cover_url" alt="" class="slot-image" />
          <p>{{ game.name }}</p>
          <button @click="removeGame(index)">Remove</button>
        </div>
        <div v-else class="empty-slot">Slot {{ index + 1 }}</div>
      </div>
    </section>

</template>

<script setup>
import { ref, watch } from "vue";

const props = defineProps({
    modelValue: Array
});

const emit = defineEmits(["update:modelValue"]);

const searchQuery = ref("");
const searchResults = ref([]);
const selectedGames = ref(props.modelValue || [null, null, null]);

watch(() => props.modelValue, (val) => {
    selectedGames.value = val || [null, null, null];
});

async function onSearchInput() {
  const q = searchQuery.value.trim();
  if (!q) {
    searchResults.value = [];
    return;
  }

  try {
    const res = await fetch(`/api/search?query=${encodeURIComponent(q)}`);
    const data = await res.json();
    searchResults.value = data.slice(0, 10);
  } catch (err) {
    console.error("Search failed:", err);
    searchResults.value = [];
  }
}

function selectGame(game) {
  const emptyIndex = selectedGames.value.findIndex((g) => !g);
  if (emptyIndex === -1) return alert("Max 3 games selected");
  selectedGames.value[emptyIndex] = game;
  searchQuery.value = "";
  searchResults.value = [];
  emit("update:modelValue", selectedGames.value);
}

function removeGame(index) {
  selectedGames.value[index] = null;
  emit("update:modelValue", selectedGames.value);
}

function getYear(dateStr) {
  if (!dateStr) return "N/A";
  const d = new Date(dateStr);
  return isNaN(d.getFullYear()) ? "N/A" : d.getFullYear();
}
</script>

<style>
.search-section {
  position: relative;
}

.search-input {
  width: 100%;
  padding: 8px;
  font-size: 1.1em;
  box-sizing: border-box;
  color: #000;
  background-color: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.suggestions-list {
  position: absolute;
  top: 38px;
  width: 100%;
  border: 1px solid #aaa;
  background: #fff;
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 200px;
  overflow-y: auto;
  z-index: 10;
  color: #000;
}

.suggestion-item {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  cursor: pointer;
  color: #000;
}

.suggestion-item:hover {
  background-color: #eef;
}

.suggestion-image {
  width: 40px;
  height: 40px;
  margin-right: 10px;
  object-fit: cover;
  border-radius: 4px;
}

.selected-games {
  display: flex;
  justify-content: space-between;
  margin: 20px 0;
}

.game-slot {
  width: 30%;
  border: 2px dashed #ccc;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 8px;
  border-radius: 8px;
  position: relative;
  color: #000;
}

.empty-slot {
  color: #999;
  font-size: 1.1em;
}

.slot-image {
  width: 100%;
  max-height: 100px;
  object-fit: contain;
  margin-bottom: 8px;
}

</style>