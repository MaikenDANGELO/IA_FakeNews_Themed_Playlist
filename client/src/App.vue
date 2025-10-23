<script setup>
import { ofetch } from 'ofetch';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import { ref } from 'vue';


const userText = ref('')
const textBoxRef = ref(null)

async function enterUserText() {
  if (textBoxRef.value && userText.value != '') {
    const newUserTextDiv = document.createElement('div')
    newUserTextDiv.classList.add('user-text')
    newUserTextDiv.setAttribute('data-v-7a7a37b1', '')

    const textP = document.createElement('p')
    textP.textContent = userText.value

    newUserTextDiv.appendChild(textP)
    textBoxRef.value.appendChild(newUserTextDiv)
    userText.value = ''

    let AIResponse = "";
    try{
      AIResponse = await ofetch(`http://localhost:8000/`, {
      method: 'GET'
    })
    }catch(e){
      console.log(e);
    }
    
    if(AIResponse == null || AIResponse == "") AIResponse = "Erreur : Impossible de communiquer avec l'IA";

    addAIText(AIResponse);
  }

  function addAIText(AItext) {
    if (textBoxRef.value) {
      const newAITextDiv = document.createElement('div');
      newAITextDiv.classList.add('ai-text');
      newAITextDiv.setAttribute('data-v-7a7a37b1', '');

      const textP = document.createElement('p');
      textP.textContent = AItext;

      newAITextDiv.appendChild(textP);
      textBoxRef.value.appendChild(newAITextDiv);
      userText.value = '';
    }
  }
}
</script>

<template>
  <div class="flex flex-col w-screen h-screen bg-black justify-center gap-1 text-white">
    <h1 class="text-5xl self-center mb-5">Fake News Viber</h1>
    <div
      id="text-box"
      ref="textBoxRef"
      class="flex flex-col bg-gray-900 w-[50%] h-[80%] self-center rounded-md border-2 border-[#34d399] p-5 gap-5 overflow-y-auto"
    ></div>
    <div class="flex flex-row w-[50%] h-[5%] self-center rounded-md gap-2 justify-center">
      <InputText
        v-model="userText"
        placeholder="Écrire..."
        class="w-full h-[66%] self-center"
        @keyup.enter="enterUserText()"
      />
      <Button icon="pi pi-send" class="h-[66%] self-center" @click="enterUserText()" />
    </div>
  </div>
</template>

<style scoped>
.ai-text {
  align-self: self-start;
  background-color: var(--color-gray-950);
  padding: 10px;
  border-radius: var(--radius-md);
}

.user-text {
  align-self: self-end;
  background-color: var(--color-gray-950);
  padding: 10px;
  border-radius: var(--radius-md);
}
</style>
