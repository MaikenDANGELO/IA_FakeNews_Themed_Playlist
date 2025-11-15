<script setup>
import { ofetch } from 'ofetch'
import { InputText, Button, Slider } from 'primevue'
import { ref, nextTick } from 'vue'

const userText = ref('')
const isUserTextBlocked = ref(false)
const textBoxRef = ref(null)

const mediaPlayerCurrentTime = ref(0)

async function enterUserText() {
  isUserTextBlocked.value = true
  if (textBoxRef.value && userText.value != '') {
    const newUserTextDiv = document.createElement('div')
    newUserTextDiv.classList.add('user-text')
    newUserTextDiv.setAttribute('data-v-7a7a37b1', '')

    const textP = document.createElement('p')
    textP.textContent = userText.value

    newUserTextDiv.appendChild(textP)
    textBoxRef.value.appendChild(newUserTextDiv)

    await nextTick()
    scrollToBottom()

    const usTxt = userText.value
    userText.value = ''

    let AIResponse = await getAIResponse(usTxt)

    if (AIResponse == null || AIResponse == '')
      AIResponse = "Erreur : Impossible de communiquer avec l'IA"

    addAIText(AIResponse)


    isUserTextBlocked.value = false
  }
}

async function getAIResponse(text) {
  let AIResponse
  try {
    AIResponse = await ofetch(`http://localhost:8000/askAI`, {
      method: 'POST',
      body: { question: text },
    })
  } catch (e) {
    console.log(e)
  }

  return AIResponse
}

function addAIText(AItext) {
  if (textBoxRef.value) {
    const newAITextDiv = document.createElement('div')
    newAITextDiv.classList.add('ai-text')
    newAITextDiv.setAttribute('data-v-7a7a37b1', '')

    const textP = document.createElement('p')
    textP.textContent = AItext

    newAITextDiv.appendChild(textP)
    textBoxRef.value.appendChild(newAITextDiv)

    nextTick(() => {
      scrollToBottom()
    })
  }
}

function scrollToBottom() {
  if (textBoxRef.value) {
    textBoxRef.value.scrollTop = textBoxRef.value.scrollHeight
  }
}
</script>

<template>
  <div class="flex flex-row w-screen h-screen p-10 bg-black gap-2 text-white">
    <div class="w-[20vw]" />

    <div class="w-[60vw] flex flex-col">
      <h1 class="text-5xl self-center mb-5">Fake News Viber</h1>
      <div id="text-box" ref="textBoxRef"
        class="flex flex-col bg-gray-950 w-[50%] h-[80%] self-center rounded-md border-2 border-(--p-primary-color) p-5 gap-5 overflow-y-auto">
      </div>

      <div class="flex flex-row w-[50%] h-[5%] self-center rounded-md gap-2 justify-center">
        <InputText v-model="userText" placeholder="Écrire..." class="w-full h-[66%] self-center"
          @keyup.enter="enterUserText()" :disabled="isUserTextBlocked" />
        <Button icon="pi pi-send" class="h-[66%] self-center" @click="enterUserText()" :disabled="isUserTextBlocked" />
      </div>
    </div>

    <div class="w-[20vw] flex flex-col">
      <h1 class="text-5xl self-center mb-5">Playlist</h1>
      <div class="bg-gray-950 w-full h-[80%] self-center rounded-md border-2 border-(--p-primary-color)"></div>
      <div class="flex flex-row gap-5 mt-5">
        <Button icon="pi pi-step-backward" />
        <Button icon="pi pi-pause" />
        <Button icon="pi pi-caret-right" />
        <Button icon="pi pi-step-forward" />
        <Slider v-model="mediaPlayerCurrentTime" class="w-[80%] self-center" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.ai-text {
  align-self: self-start;
  background-color: var(--color-gray-900);
  padding: 10px;
  border-radius: var(--radius-md);
  max-width: 60%;
  font-style: italic;
  word-wrap: break-word;
}

.user-text {
  align-self: self-end;
  background-color: var(--color-gray-700);
  padding: 10px;
  border-radius: var(--radius-md);
  max-width: 60%;
  word-wrap: break-word;
}
</style>
