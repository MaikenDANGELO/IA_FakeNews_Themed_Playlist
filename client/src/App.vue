<script setup>
import { InputText, Button, Slider } from 'primevue'
import { ref, nextTick, watch } from 'vue'

const userText = ref('')
const isUserTextBlocked = ref(false)
const textBoxRef = ref(null)
const mediaPlayerCurrentTime = ref(0)

// We store the conversation history here
// Structure: { role: 'user' | 'ai', text: string }
const messages = ref([])

async function enterUserText() {
  if (!userText.value.trim()) return

  isUserTextBlocked.value = true
  const currentQuestion = userText.value

  messages.value.push({
    role: 'user',
    text: currentQuestion,
  })

  userText.value = ''
  await nextTick()
  scrollToBottom()

  // --- CORRECTION ICI ---

  // 1. On pousse un objet vide
  messages.value.push({
    role: 'ai',
    text: '',
  })

  // 2. IMPORTANT : On récupère l'objet qui est DANS le tableau (c'est lui le Proxy Réactif)
  const reactiveMessageObject = messages.value[messages.value.length - 1]

  // 3. On passe cet objet réactif à la fonction de stream
  await streamAIResponse(currentQuestion, reactiveMessageObject)

  isUserTextBlocked.value = false
}

async function streamAIResponse(question, messageObject) {
  try {
    const response = await fetch(`http://localhost:8000/askAI`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question }),
    })

    if (!response.body) return

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })

      // Update the reactive object. Vue sees this change and updates the DOM.
      messageObject.text += chunk

      await nextTick();
      scrollToBottom()
    }
  } catch (e) {
    console.error(e)
    messageObject.text += ' [Error: AI unreachable]'
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

      <div
        ref="textBoxRef"
        class="flex flex-col bg-gray-950 w-[50%] h-[80%] self-center rounded-md border-2 border-(--p-primary-color) p-5 gap-5 overflow-y-auto"
      >
        <div
          v-for="(msg, index) in messages"
          :key="index"
          :class="msg.role === 'user' ? 'user-text' : 'ai-text'"
        >
          <i v-if="msg.role !== 'user' && !msg.text" class="pi pi-ellipsis-h" />
          <p>{{ msg.text }}</p>
        </div>
      </div>

      <div class="flex flex-row w-[50%] h-[5%] self-center rounded-md gap-2 justify-center mt-2">
        <InputText
          v-model="userText"
          placeholder="Écrire..."
          class="w-full h-[66%] self-center"
          @keyup.enter="enterUserText()"
          :disabled="isUserTextBlocked"
        />
        <Button
          icon="pi pi-send"
          class="h-[66%] self-center"
          @click="enterUserText()"
          :disabled="isUserTextBlocked"
        />
      </div>
    </div>

    <div class="w-[20vw] flex flex-col">
      <h1 class="text-5xl self-center mb-5">Playlist</h1>
      <div
        class="bg-gray-950 w-full h-[80%] self-center rounded-md border-2 border-(--p-primary-color)"
      ></div>
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
/* Note: I added a transition for smoother appearance if you like */
.ai-text,
.user-text {
  transition: all 0.2s ease;
}

.ai-text {
  align-self: self-start;
  background-color: var(--p-surface-900); /* Using PrimeVue variable or your hardcoded hex */
  background-color: #1f2937; /* Fallback to gray-900 equivalent */
  padding: 10px;
  border-radius: 10px;
  max-width: 60%;
  font-style: italic;
  word-wrap: break-word;
}

.user-text {
  align-self: self-end;
  background-color: var(--p-surface-700);
  background-color: #374151; /* Fallback to gray-700 equivalent */
  padding: 10px;
  border-radius: 10px;
  max-width: 60%;
  word-wrap: break-word;
}
</style>
