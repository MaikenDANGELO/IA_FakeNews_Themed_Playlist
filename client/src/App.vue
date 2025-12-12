<script setup>
import { Textarea, Button } from 'primevue'
import { ref, nextTick } from 'vue'

const userText = ref('')
const isUserTextBlocked = ref(false)
const textBoxRef = ref(null)

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

  messages.value.push({
    role: 'ai',
    text: '',
  })

  const reactiveMessageObject = messages.value[messages.value.length - 1]

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
  <div class="flex flex-row w-screen h-screen p-10 bg-[#050505] gap-2 text-white">
    <div class="w-screen flex flex-col">
      <h1 class="text-5xl self-center mb-5">Fake News Viber</h1>

      <div
        ref="textBoxRef"
        class="flex flex-col bg-gray-950 w-full h-[80%] self-center rounded-md border-2 border-(--p-primary-color) p-5 gap-5 overflow-y-auto"
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

      <div class="flex flex-row w-full min-h-[5%] self-center rounded-md gap-2 justify-center mt-2">
        <Textarea
          v-model="userText"
          placeholder="Écrire..."
          class="w-full self-center bg-gray-950 border-2 border-(--p-primary-color) rounded-md p-2"
          @keyup.enter="enterUserText()"
          :disabled="isUserTextBlocked"
          auto-resize
        />
        <Button
          icon="pi pi-send"
          class="h-[66%] self-center"
          @click="enterUserText()"
          :disabled="isUserTextBlocked"
        />
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
  border: 1px solid violet;
}

.user-text {
  align-self: self-end;
  background-color: var(--p-surface-700);
  background-color: #374151; /* Fallback to gray-700 equivalent */
  padding: 10px;
  border-radius: 10px;
  max-width: 60%;
  word-wrap: break-word;
  border: 1px solid goldenrod;
}
</style>
