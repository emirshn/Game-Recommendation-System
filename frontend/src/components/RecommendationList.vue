<template>
  <section class="recommendations" v-if="recommendations.length">
    <h2>Recommendations</h2>
    <div v-for="game in pagedRecommendations" :key="game.id" class="recommendation-card">
      <!-- Game card -->
      <img :src="game.cover_url" alt="Cover" class="recommend-cover-tall" />
      <div class="recommend-info">
        <div class="recommend-header">
          <h3>{{ game.name }}</h3>
          <div class="score-boxes">
            <span class="score user" :class="scoreColor(game.user_score)">
              User: {{ validScore(game.user_score) ? formatScore(game.user_score) : "N/A" }}
            </span>
            <span class="score critic" :class="scoreColor(game.critic_score)">
              Critic: {{ validScore(game.critic_score) ? formatScore(game.critic_score) : "N/A" }}
            </span>
          </div>
        </div>

        <p><strong>Release Date:</strong> {{ formatDate(game.release_date) }}</p>
        <p><strong>Genres:</strong> {{ game.genres }}</p>
        <p><strong>Themes:</strong> {{ game.themes }}</p>
        <p><strong>Developers:</strong> {{ game.main_developers }}</p>
        <p><strong>Series:</strong> {{ game.series }}</p>
        <p><strong>Platforms:</strong> {{ game.platforms }}</p>

        <button @click="toggleDetails(game)" class="toggle-details">
          {{ game.showDetails ? "Hide" : "Show" }} Details
        </button>

        <div v-if="game.showDetails" class="game-details">
          <p><strong>Summary:</strong> {{ game.summary }}</p>
          <p><strong>Franchise:</strong> {{ game.franchise }}</p>
          <p><strong>Supporting Developers:</strong> {{ game.supporting_developers }}</p>
          <p><strong>Publishers:</strong> {{ game.publishers }}</p>
          <p><strong>Player Perspectives:</strong> {{ game.player_perspectives }}</p>
          <p><strong>Game Modes:</strong> {{ game.game_modes }}</p>
          <p><strong>Game Engines:</strong> {{ game.game_engines }}</p>

          <div v-if="game.keywords" class="keywords-container">
            <strong>Keywords:</strong>
            <div class="keyword-tags">
              <span
                v-for="tag in getKeywordTags(game.keywords)"
                :key="tag"
                class="keyword-tag"
                :style="{ backgroundColor: getTagColor(tag) }"
              >
                {{ tag }}
              </span>
            </div>
          </div>

          <div v-if="game.screenshot_urls.length" class="screenshots-container">
            <h4>Screenshots</h4>
            <div class="screenshots">
              <img
                v-for="(url, i) in game.screenshot_urls"
                :src="url"
                :key="url"
                class="screenshot"
                @click.stop="openZoomGallery(game.screenshot_urls, i)"
              />
            </div>
          </div>

          <div v-if="game.artwork_urls.length" class="screenshots-container">
            <h4>Artwork</h4>
            <div class="screenshots">
              <img
                v-for="(url, i) in game.artwork_urls"
                :src="url"
                :key="url"
                class="screenshot"
                @click.stop="openZoomGallery(game.artwork_urls, i)"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
  <section class="no-results" v-else-if="recommendations.length === 0 && originalRecommendations && originalRecommendations.length > 0">
  <div class="no-results-card">
    <img src="/no-results.png" alt="No results" class="no-results-image" />
    <p>No games match your filter</p>
  </div>
  </section>

  <div class="pagination" v-if="totalPages > 1">
    <button :disabled="currentPage === 1" @click="currentPage--" class="page-btn">Prev</button>
    <button
      v-for="page in totalPages"
      :key="page"
      :class="['page-btn', { active: page === currentPage }]"
      @click="currentPage = page"
    >
      {{ page }}
    </button>
    <button :disabled="currentPage === totalPages" @click="currentPage++" class="page-btn">Next</button>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  recommendations: Array,
  originalRecommendations: Array
});

const emit = defineEmits(['open-zoom']);

const currentPage = ref(1);
const itemsPerPage = 10;

const pagedRecommendations = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage;
  return props.recommendations.slice(start, start + itemsPerPage);
});

const totalPages = computed(() =>
  Math.ceil(props.recommendations.length / itemsPerPage)
);

function formatDate(dateStr) {
  if (!dateStr) return "N/A";
  const d = new Date(dateStr);
  return isNaN(d) ? "N/A" : d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric"
  });
}

function validScore(score) {
  const n = Number(score);
  return !isNaN(n) && n >= 0 && n <= 100;
}

function formatScore(score) {
  let n = Number(score);
  return isNaN(n) ? 'N/A' : (n >= 100 ? '100' : n.toFixed(0));
}

function scoreColor(score) {
  let n = Number(score);
  if (isNaN(n)) return 'score-gray';
  if (n >= 80) return 'score-green';
  if (n >= 60) return 'score-yellow';
  if (n >= 40) return 'score-orange';
  return 'score-red';
}

const tagColors = [
  "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
  "#e74c3c", "#1abc9c", "#f1c40f",
];

function getTagColor(tag) {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) {
    hash += tag.charCodeAt(i);
  }
  return tagColors[hash % tagColors.length];
}

function getKeywordTags(keywordStr) {
  if (!keywordStr) return [];
  return keywordStr.split(',').map(k => k.trim()).filter(Boolean);
}

function toggleDetails(game) {
  game.showDetails = !game.showDetails;
}

function openZoomGallery(urls, index) {
  emit('open-zoom', { urls, index });
}
</script>

<style scoped>
.pagination {
  margin: 1em 0;
  text-align: center;
}

.page-btn {
  margin: 0 0.25em;
  padding: 0.3em 0.7em;
  border: 1px solid #888;
  background-color: white;
  cursor: pointer;
  color: rgb(0, 0, 0);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-btn.active {
  background-color: #007bff;
  color: rgb(255, 255, 255);
  border-color: #007bff;
}

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

.recommendations {
  margin-top: 30px;
  color: #000;
}

.recommendation-card {
  display: flex;
  flex-direction: row;
  gap: 16px;
  margin-bottom: 20px;
  background-color: #ffffff;
  padding: 16px;
  border-radius: 12px;
  flex-wrap: nowrap;
  align-items: flex-start;

  border: 1px solid #ccc;  
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.recommend-cover-tall {
  width: 150px;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
  flex-shrink: 0;
}

.recommend-info {
  flex: 1;
  min-width: 250px;
}

.recommend-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 8px;
}

.recommend-info h3 {
  margin: 0;
  font-size: 1.4em;
  color: #000;
  text-align: center;
}
.score-box {
  width: 50px;
  height: 50px;
  border-radius: 8px;
  font-weight: bold;
  font-size: 16px;
  color: #fff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.score-box .label {
  font-size: 10px;
  margin-top: 2px;
  color: #eee;
}

.score {
  font-weight: bold;
  font-size: 1.2rem; 
  color: white;
  padding: 8px 14px;
  border-radius: 8px;
  min-width: 48px;
  text-align: center;
  user-select: none;
  display: inline-block;
  box-sizing: border-box;
  white-space: nowrap;
}

.score-gray {
  background-color: gray;
}

.score-green {
  background-color: #4caf50;
}

.score-yellow {
  background-color: #ffc107;
  color: #000; 
}

.score-orange {
  background-color: #ff9800;
}

.score-red {
  background-color: #f44336;
}

.score-boxes {
  display: flex;
  gap: 12px;
  justify-content: center; 
  align-items: center;
  margin-top: 8px;
}

.toggle-details {
  margin-top: 8px;
  background-color: #007acc;
  color: white;
  border: none;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
  width: fit-content;
}

.game-details {
  margin-top: 10px;
  max-width: 100%;
}

.screenshots-container {
  margin-top: 10px;
}

.screenshots-container h4 {
  margin-bottom: 6px;
  color: #333;
  font-weight: 600;
  user-select: none;
}

.screenshots {
  overflow-x: auto;
  white-space: nowrap;
  padding-bottom: 8px;
}

.screenshots img {
  display: inline-block;
  height: 120px;
  margin-right: 10px;
  border-radius: 6px;
  object-fit: cover;
}
.no-results {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 30px;
  color: #444;
}

.no-results-card {
  background-color: #f9f9f9;
  padding: 24px;
  border-radius: 12px;
  border: 1px solid #ccc;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  max-width: 400px;
}

.no-results-image {
  width: 120px;
  height: auto;
  margin-bottom: 16px;
  opacity: 0.7;
}

</style>