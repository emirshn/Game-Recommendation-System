<template>
  <section class="recommend-filters">
    <div class="filters-row">
      <label>
        Sort:
        <select v-model="localSortOption">
          <option value="">None</option>
          <option value="user">User Score</option>
          <option value="critic">Critic Score</option>
          <option value="date">Release Date</option>
          <option value="score_rec">Recommendation Score</option>
        </select>
      </label>

      <label>
        Direction:
        <select v-model="localSortDirection">
          <option value="desc">Descending</option>
          <option value="asc">Ascending</option>
        </select>
      </label>

      <label>
        Min User Score:
        <input type="number" v-model.number="localMinUserScore" min="0" max="100" />
      </label>

      <label>
        Min Critic Score:
        <input type="number" v-model.number="localMinCriticScore" min="0" max="100" />
      </label>

      <label>
        After Year:
        <input type="number" v-model.number="localMinYear" placeholder="e.g. 2015" />
      </label>

      <label>
        <input type="checkbox" v-model="localHideUnreleased" />
        Hide unreleased games
      </label>

      <label>
        <input type="checkbox" v-model="localHideZeroScore" />
        Hide games with no user or critic score
      </label>

      <div class="apply-button-wrapper">
        <button class="apply-button" @click="emitApply">Apply</button>
      </div>
    </div>
  </section>
</template>

<script setup>
const props = defineProps({
  sortOption: String,
  sortDirection: String,
  minUserScore: Number,
  minCriticScore: Number,
  minYear: Number,
  hideUnreleased: { type: Boolean, default: true },
  hideZeroScore: { type: Boolean, default: true }
});

const emit = defineEmits([
  'update:sortOption',
  'update:sortDirection',
  'update:minUserScore',
  'update:minCriticScore',
  'update:minYear',
  'update:hideUnreleased',
  'update:hideZeroScore',
  'apply'
]);

const localSortOption = ref(props.sortOption);
const localSortDirection = ref(props.sortDirection);
const localMinUserScore = ref(props.minUserScore);
const localMinCriticScore = ref(props.minCriticScore);
const localMinYear = ref(props.minYear);
const localHideUnreleased = ref(props.hideUnreleased);
const localHideZeroScore = ref(props.hideZeroScore);

watch(localSortOption, val => emit('update:sortOption', val));
watch(localSortDirection, val => emit('update:sortDirection', val));
watch(localMinUserScore, val => emit('update:minUserScore', val));
watch(localMinCriticScore, val => emit('update:minCriticScore', val));
watch(localMinYear, val => emit('update:minYear', val));
watch(localHideUnreleased, val => emit('update:hideUnreleased', val));
watch(localHideZeroScore, val => emit('update:hideZeroScore', val));

import { ref, watch } from 'vue';

const emitApply = () => emit('apply');
</script>

<style scoped>
.recommend-filters {
  background-color: #fff;
  color: #000;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ccc; 
}
.recommend-filters label {
  color: #000;
}
.recommend-filters select,
.recommend-filters input[type="number"] {
  background-color: #fff;
  color: #000;
  border: 1px solid #ccc;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 14px;
  appearance: none;
}
.filters-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
}

.apply-button-wrapper {
  flex-basis: 100%;
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
.filters-row label {
  font-size: 0.95em;
  color: #333;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 120px;
}

.filters-row input,
.filters-row select {
  padding: 6px 8px;
  font-size: 0.95em;
  border-radius: 4px;
  border: 1px solid #ccc;
}

.apply-button {
  padding: 8px 14px;
  font-size: 1em;
  background-color: #007acc;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.recommend-filters input[type="checkbox"] {
  transform: scale(1.1);
  margin-right: 6px;
}
</style>
