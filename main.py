from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load the pre-trained model and tokenizer
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=AutoTokenizer.from_pretrained(model_name))

# Load the transcript from a .txt file
file_path = "videos.txt"
with open(file_path, "r", encoding="utf-8") as file:
    transcript = file.read()

# Split long transcript into smaller chunks
def split_transcript(transcript, max_length=500):
    words = transcript.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

chunks = split_transcript(transcript)

# Filter chunks based on word matching
def get_relevant_chunks(question, chunks):
    question_words = set(question.lower().split())  # Split question into lowercase words
    relevant_chunks = [
        chunk for chunk in chunks if any(word in chunk.lower() for word in question_words)
    ]
    return relevant_chunks if relevant_chunks else chunks  # Return all chunks if no matches found

# Combine answers from multiple chunks
def answer_question_combined(question, chunks):
    answers = []
    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result['score'] > 0.5:  # Only consider confident answers
            answers.append((result['answer'], result['score']))

    if answers:
        # Sort answers by score and return the best one
        sorted_answers = sorted(answers, key=lambda x: x[1], reverse=True)
        return sorted_answers[0][0], sorted_answers[0][1]
    else:
        return "Sorry, I couldn't find a clear answer.", 0

# Chatbot interaction
print("Chatbot: Hi! Ask me anything about the video or type 'quit' to exit.")
while True:
    question = input("You: ")
    if question.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Thanks for chatting! Goodbye!")
        break

    # Get relevant chunks
    relevant_chunks = get_relevant_chunks(question, chunks)

    # Get the answer from relevant chunks
    answer, confidence = answer_question_combined(question, relevant_chunks)

    # Respond to the user
    if confidence > 0.8:
        print(f"Chatbot: Here's what I found - {answer}. (Confidence: {confidence:.2f})")
    elif confidence > 0.5:
        print(f"Chatbot: I think the answer is {answer}, but I'm not 100% sure.")
    else:
        print("Chatbot: Sorry, I couldn't find a clear answer in the transcript. Could you rephrase?")
