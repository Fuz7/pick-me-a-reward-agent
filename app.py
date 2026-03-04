import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json
import uuid  # for unique keys


llm = FakeLLM()
chat_box = ChatBox(
    use_rich_markdown=True, # use streamlit-markdown
    user_theme="green", # see streamlit_markdown.st_markdown for all available themes
    assistant_theme="blue",
    
)
chat_box.use_chat_name("chat1") # add a chat conversatoin

def on_chat_change():
    chat_box.use_chat_name(st.session_state["chat_name"])
    chat_box.context_to_session() # restore widget values to st.session_state when chat name changed

in_expander = True

with st.sidebar:
    st.markdown(
    "<h1 style='text-align: center; font-weight: bold; font-size: 50px;'>Earn Your Chill</h1>",
    unsafe_allow_html=True
)

    streaming = False
    chat_box.context_from_session(exclude=["chat_name"]) # save widget values to chat context
    st.divider()
    import streamlit as st

st.sidebar.markdown("""
    <!-- Title -->


    <!-- Description -->
    <p style="
        font-size:14px; 
        line-height:1.6; 
        color:#333; 
        font-family: 'Verdana', sans-serif; 
        margin-top:0;
    ">
    Your chill, your rules! 😎 Choose <strong>how long</strong> you want to relax, set your <strong>reward</strong>, and even track how long <strong>to chill</strong> before you indulge.  
    <br><br>
    <span style="
        font-weight:bold; 
        color:#4169E1; 
        letter-spacing:1px;
    ">Earn Your Chill</span> makes productivity personal — the more effort you put in, the better your reward. <span style="color:#2196F3;">🎮☕</span>
    </p>
""", unsafe_allow_html=True)
chat_box.init_session()
chat_box.output_messages()

def on_feedback(
    feedback,
    chat_history_id: str = "",
    history_index: int = -1,
):
    reason = feedback["text"]
    score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index) # convert emoji to integer
    # do something
    st.session_state["need_rerun"] = True


feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "wellcome to feedback",
}

st.markdown("""
<style>
/* Make chat_input focus blue using :focus-within */
.st-emotion-cache-6mn6c9:focus-within {
    outline: none !important;                /* remove default red focus */
    border: 2px solid #2196F3 !important;   /* blue border */
    box-shadow: 0 0 3px #2196F3 !important; /* subtle blue glow */
}
            /* Chat submit button */
button[data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background-color: #2196F3 !important; /* blue background */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

/* Hover effect for submit button */

</style>
""", unsafe_allow_html=True)


if query := st.chat_input('input your question here'):
    chat_box.user_say(query)
    if streaming:
        generator = llm.chat_stream(query)
        elements = chat_box.ai_say(
            [
                # you can use string for Markdown output if no other parameters provided
                Markdown("thinking", in_expander=in_expander,
                         expanded=True, title="answer"),
                Markdown("", in_expander=in_expander, title="references"),
            ]
        )
        time.sleep(0.1)
        text = ""
        for x, docs in generator:
            text += x
            chat_box.update_msg(text, element_index=0, streaming=True)
        # update the element without focus
        chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
        chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
        chat_history_id = f"feedback_{uuid.uuid4()}"
        chat_box.show_feedback(**feedback_kwargs,
                                key=chat_history_id,
                                on_submit=on_feedback,
                                kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
    else:
        text, docs = llm.chat(query)
        chat_box.ai_say(
            [
                Markdown(text, in_expander=in_expander,
                         expanded=True, title="answer"),
                Markdown("\n\n".join(docs), in_expander=in_expander,
                         title="references"),
            ]
        )


cols = st.columns(1)

if cols[0].button('run agent'):
    chat_box.user_say('run agent')
    agent = FakeAgent()
    text = ""

    # streaming:
    chat_box.ai_say() # generate a blank placeholder to render messages
    for d in agent.run_stream():
        if d["type"] == "complete":
            chat_box.update_msg(expanded=False, state="complete")
            chat_box.insert_msg(d["llm_output"])
            break

        if d["status"] == 1:
            chat_box.update_msg(expanded=False, state="complete")
            text = ""
            chat_box.insert_msg(Markdown(text, title=d["text"], in_expander=True, expanded=True))
        elif d["status"] == 2:
            text += d["llm_output"]
            chat_box.update_msg(text, streaming=True)
        else:
            chat_box.update_msg(text, streaming=False)



