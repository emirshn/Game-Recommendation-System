<template>
  <header class="header">
    <div class="logo-container">
      <span class="logo-text">Game Recommendation System</span> 
    </div>
    <p class="description">
      Search for games, pick up to 3, set options, and get personalized recommendations.
    </p>
  </header>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

.header {
  text-align: center;
  margin-bottom: 20px;
}

.logo-container {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px; 
  margin-bottom: 8px;
}

.logo {
  max-width: 60px; 
  height: auto;
}

.logo-text {
  font-family: 'Press Start 2P', cursive;
  font-weight: normal;
  font-size: 1.2rem; 
  color: #00ccff; 
  letter-spacing: 0.1em; 
  user-select: none;
  text-shadow:
    1px 1px 0 #000,
    2px 2px 0 #007acc;
}

  
.description {
  font-size: 1.1em;
  color: #555;
}

</style>
