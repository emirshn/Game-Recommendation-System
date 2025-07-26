<template>
  <section class="recommend-section">
    <button
      :disabled="disabled"
      class="recommend-button"
    >
      Recommend
    </button>

    <div v-if="loading" class="spinner"></div>
  </section>
</template>

<script setup>
const props = defineProps({
  disabled: { type: Boolean, default: false },
  loading: { type: Boolean, default: false },
});
</script>

<style scoped>
.recommend-section {
  text-align: center;
  margin-bottom: 20px;
}

.recommend-button {
  background: #28a745;
  color: white;
  border: none;
  font-size: 1.2em;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
}

.recommend-button:disabled {
  background: #aaa;
  cursor: not-allowed;
}

.spinner {
  margin: 12px auto 0;
  border: 4px solid #eee;
  border-top: 4px solid #28a745;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

</style>
