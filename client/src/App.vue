<script setup>
import InputText from 'primevue/inputtext'
import Button from 'primevue/button'
</script>

<template>
  <div class="flex flex-col w-screen h-screen bg-black justify-center gap-1 text-white">
    <h1 class="text-5xl self-center mb-5">Fake News Viber</h1>
    <div class="bg-gray-900 w-[50%] h-[80%] self-center rounded-md border-2 border-[#34d399]">

    </div>
    <div class="flex flex-row w-[50%] h-[5%] self-center rounded-md gap-2 justify-center">
      <InputText placeholder="Écrire..." class="w-full h-[66%] self-center" />
      <Button icon="pi pi-send" class="h-[66%] self-center" />
    </div>
  </div>
</template>

<style scoped>

</style>
