<template>
  <div class="app-container">
    <HeaderSection/>

    <GameSelector v-model="selectedGames" />
    <div class="filter-tip">
      <strong>Tip:</strong> Use the advanced options below to reduce noise and focus your recommendations.
      Setting filters like include and exclude keywords, genres, themes etc. can help get more accurate and relevant result.
    </div>
    <!-- Advanced options -->
    <section class="advanced-options">
      <button @click="showAdvanced = !showAdvanced" class="toggle-button">
        {{ showAdvanced ? 'Hide' : 'Show' }} Advanced Options
      </button>
      <div v-if="showAdvanced" class="options-panel">
        <label>
          Number of recommendations:
          <input type="number" v-model.number="options.limit" placeholder="e.g. 20" min="1" />
        </label>

        <label>Include Keywords:</label>
        <Multiselect
          v-model="options.includeKeywords"
          :options="allKeywords"
          placeholder="Select keywords"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>Exclude Keywords:</label>
        <Multiselect
          v-model="options.excludeKeywords"
          :options="allKeywords"
          placeholder="Select keywords"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>Include Genres:</label>
        <Multiselect
          v-model="options.includeGenres"
          :options="allGenres"
          placeholder="Select genres"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>Exclude Genres:</label>
        <Multiselect
          v-model="options.excludeGenres"
          :options="allGenres"
          placeholder="Select genres"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>Include Themes:</label>
        <Multiselect
          v-model="options.includeThemes"
          :options="allThemes"
          placeholder="Select themes"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>Exclude Themes:</label>
        <Multiselect
          v-model="options.excludeThemes"
          :options="allThemes"
          placeholder="Select themes"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />

        <label>
          Exclude games before year:
          <input type="number" v-model.number="options.excludeBeforeYear" placeholder="e.g. 2000" />
        </label>

        <label>Preferred Platforms:</label>
        <Multiselect
          v-model="options.platforms"
          :options="allPlatforms"
          placeholder="Select platforms"
          multiple
          searchable
          close-on-select="false"
          :clear-on-select="false"
        />
      </div>
    </section>

    <RecommendButton
      :disabled="isDisabled"
      :loading="loading"
      @click="recommend"
    />

    <RecommendationFilters
      v-if="originalRecommendations.length"
      :sortOption="sortOption"
      :sortDirection="sortDirection"
      :minUserScore="minUserScore"
      :minCriticScore="minCriticScore"
      :minYear="minYear"
      :hideUnreleased="hideUnreleased"
      :hideZeroScore="hideZeroScore"
      @update:sortOption="sortOption = $event"
      @update:sortDirection="sortDirection = $event"
      @update:minUserScore="minUserScore = $event"
      @update:minCriticScore="minCriticScore = $event"
      @update:minYear="minYear = $event"
      @update:hideUnreleased="hideUnreleased = $event"
      @update:hideZeroScore="hideZeroScore = $event"
      @apply="applyFilters"
    />

    <RecommendationList
      :recommendations="recommendations"
      :originalRecommendations="originalRecommendations"
      @open-zoom="({ urls, index }) => openZoomGallery(urls, index)"
    />

    <div v-if="zoomedImage" class="zoom-modal" @click.self="closeZoom">
      <button class="nav-button left" @click.stop="prevImage">←</button>

      <img :src="zoomedImage" class="zoomed-image" />

      <button class="nav-button right" @click.stop="nextImage">→</button>
      <button class="close-button" @click="closeZoom">×</button>
    </div>

  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onBeforeUnmount } from "vue";
import Multiselect from 'vue-multiselect'
import 'vue-multiselect/dist/vue-multiselect.min.css'
components: { Multiselect } 
import HeaderSection from './components/Header.vue'
import GameSelector from "./components/GameSelector.vue";
import RecommendButton from './components/RecommendButton.vue'
import RecommendationFilters from './components/RecommendFilters.vue'; 
import RecommendationList from './components/RecommendationList.vue';


const isDisabled = computed(() => {
  return selectedGames.value.filter(Boolean).length === 0 || loading.value
})

const allKeywords = ref([]);
const allGenres = ref([]);
const allPlatforms = ref([]);
const allThemes = ref([]);

onMounted(async () => {
  try {
    const res = await fetch("/api/options"); 
    const data = await res.json();
    allKeywords.value = data.keywords || [];
    allGenres.value = data.genres || [];
    allPlatforms.value = data.platforms || [];
    allThemes.value = data.themes || [];
  } catch (e) {
    console.error("Failed to fetch options", e);
  }
});

const loading = ref(false);

const currentPage = ref(1);

const zoomedImage = ref(null);

function openZoom(src) {
  zoomedImage.value = src;
}

const zoomGallery = reactive({
  urls: [],
  index: 0,
});

function openZoomGallery(urls, index = 0) {
  zoomGallery.urls = urls;
  zoomGallery.index = index;
  zoomedImage.value = urls[index];
}

function nextImage() {
  if (zoomGallery.urls.length === 0) return;
  zoomGallery.index = (zoomGallery.index + 1) % zoomGallery.urls.length;
  zoomedImage.value = zoomGallery.urls[zoomGallery.index];
}

function prevImage() {
  if (zoomGallery.urls.length === 0) return;
  zoomGallery.index =
    (zoomGallery.index - 1 + zoomGallery.urls.length) % zoomGallery.urls.length;
  zoomedImage.value = zoomGallery.urls[zoomGallery.index];
}

function closeZoom() {
  zoomedImage.value = null;
  zoomGallery.urls = [];
  zoomGallery.index = 0;
}

function onImageClick(e) {
  const target = e.target;
  if (target.tagName === "IMG") {
    openZoom(target.src);
  }
}

onMounted(() => {
  document.addEventListener("click", onImageClick);
});

onBeforeUnmount(() => {
  document.removeEventListener("click", onImageClick);
});

const options = reactive({
  excludeBeforeYear: null,
  limit: 50,
  excludeKeywords: [],
  includeKeywords: [],        
  includeGenres: [],          
  excludeGenres: [],          
  includeThemes: [],
  excludeThemes: [],
  platforms: [],              
});

const selectedGames = ref([null, null, null]);
const recommendations = ref([]);
const showAdvanced = ref(false);

watch(recommendations, () => {
  currentPage.value = 1;
});

function isValidScore(score) {
  if (score === null || score === undefined || score === "") return false;
  const n = Number(score);
  return !isNaN(n) && n > 0;
}

function isReleased(releaseDateStr) {
  if (!releaseDateStr || releaseDateStr.trim() === "") return false; 
  const releaseDate = new Date(releaseDateStr);
  if (isNaN(releaseDate)) return false; 
  const today = new Date();
  return releaseDate <= today; 
}
const sortOption = ref(""); 
const sortDirection = ref("desc"); 
const minUserScore = ref(null);
const minCriticScore = ref(null);
const minYear = ref(null);
const hideUnreleased = ref(true);
const hideZeroScore = ref(true);

function applyFilters() {
  recommendations.value = filterRecommendations(originalRecommendations.value.map(g => ({
    ...g,
    showDetails: false,
  })));
}

function filterRecommendations(games) {
  return games
    .filter(game => {
      if (hideUnreleased.value && !isReleased(game.release_date)) return false;

      const userValid = isValidScore(game.user_score);
      const criticValid = isValidScore(game.critic_score);

      if (hideZeroScore.value && !userValid && !criticValid) return false;

      const userScore = Number(game.user_score);
      const criticScore = Number(game.critic_score);
      const year = new Date(game.release_date).getFullYear();

      if (minUserScore.value !== null && userScore < minUserScore.value) return false;
      if (minCriticScore.value !== null && criticScore < minCriticScore.value) return false;
      if (minYear.value !== null && year < minYear.value) return false;

      return true;
    })
    .sort((a, b) => {
      let valA = 0;
      let valB = 0;

      if (sortOption.value === "user") {
        valA = Number(a.user_score);
        valB = Number(b.user_score);
      } else if (sortOption.value === "critic") {
        valA = Number(a.critic_score);
        valB = Number(b.critic_score);
      } else if (sortOption.value === "date") {
        valA = new Date(a.release_date);
        valB = new Date(b.release_date);
      } else if (sortOption.value === "score_rec") {
        valA = a.rec_score != null ? Number(a.rec_score) : -Infinity;
        valB = b.rec_score != null ? Number(b.rec_score) : -Infinity;
      }

      return sortDirection.value === "asc" ? valA - valB : valB - valA;
    });
}

const originalRecommendations = ref([]);

function handleRecommend(newRecs) {
  originalRecommendations.value = newRecs;
  recommendations.value = [...newRecs]; 
}

async function recommend() {
  const selectedIds = selectedGames.value.filter(Boolean).map(g => g.id); 

  if (selectedIds.length === 0) return;
  loading.value = true;  
  try {
    const res = await fetch("http://127.0.0.1:8000/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        selected_games: selectedIds, 
        options: options,
      }),
    });

    const data = await res.json();
    console.log("Raw recommendations:", data.recommendations);
    handleRecommend(data.recommendations)
    const filtered = filterRecommendations(data.recommendations);

    recommendations.value = filtered.map(game => ({
      ...game,
      showDetails: false,
    }));
  } catch (err) {
    console.error("Recommendation request failed:", err);
    recommendations.value = [];
  } finally{
    loading.value=false;
  }
}

</script>

<style scoped>

.keyword-tag {
  display: inline-block;
  background-color: #3498db; 
  color: white;
  padding: 3px 8px;
  margin: 2px 4px 2px 0;
  border-radius: 12px;
  font-size: 0.8rem;
  user-select: none;
  cursor: default;
}
.filter-tip {
  background-color: #fffbe6;
  border-left: 5px solid #ffc107;
  padding: 12px 16px;
  margin: 16px 0;
  font-size: 14px;
  color: #333;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.app-container {
  max-width: 800px;
  margin: 20px auto;
  font-family: Arial, sans-serif;
  padding: 10px;
  color: #000;
  background-color: #fff;
}

.advanced-options {
  margin-bottom: 20px;
}

.toggle-button {
  background: #007acc;
  border: none;
  color: white;
  padding: 8px 12px;
  border-radius: 4px;
  cursor: pointer;
}

.options-panel {
  margin-top: 10px;
  border: 1px solid #ccc;
  padding: 10px;
  background: #fafafa;
  border-radius: 4px;
  color: #000;
}

.options-panel label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #000;
}

.options-panel input[type="text"],
.options-panel input[type="number"] {
  width: 100%;
  padding: 6px 8px;
  box-sizing: border-box;
  font-size: 1em;
  color: #000;
  background-color: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.zoom-modal {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000;
  background: rgba(0, 0, 0, 0.9);
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.zoomed-image {
  max-width: 80%;
  max-height: 80%;
  object-fit: contain;
  border-radius: 8px;
}

.nav-button {
  position: absolute;
  top: 50%;
  font-size: 2rem;
  background: transparent;
  color: white;
  border: none;
  cursor: pointer;
  padding: 1rem;
  z-index: 1001;
  user-select: none;
}

.nav-button.left {
  left: 2%;
}

.nav-button.right {
  right: 2%;
}

.close-button {
  position: absolute;
  top: 5%;
  right: 5%;
  font-size: 2rem;
  background: transparent;
  color: white;
  border: none;
  cursor: pointer;
  z-index: 1001;
}

</style>
