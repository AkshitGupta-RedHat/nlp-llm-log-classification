from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq = Groq()



def classify_with_llm(log_message):
    prompt = f'''CLassify the log message into one of the following categories:
                1 - User Action
                2 - System Notification
                3 - Depreciating Warning.
                If can't figure out a category, return "Unclassified".
                Only return the category name. Not preamle.
                log message: {log_message}
                '''
    chat_completion = groq.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    log_message = "User User123 logged in"
    print(classify_with_llm(log_message))

    log_message = "HI how are you"
    print(classify_with_llm(log_message))
