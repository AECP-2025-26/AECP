import streamlit as st
import datetime

# --- CSS Styling for the Homepage and Articles ---
# This CSS makes the homepage look "fancy" and ensures basic readability for articles.
# It uses Flexbox for layout and adds some visual flair.
# Note: Streamlit's internal styling might override some of these, but it provides a good base.
CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'); /* Import a nice font */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f5;
        color: #333;
        margin: 0;
        padding: 20px;
    }
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    .homepage-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 40px 30px;
        text-align: center;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        animation: fadeIn 1s ease-out;
    }
    .homepage-header h1 {
        margin-top: 0;
        font-size: 2.8em;
        margin-bottom: 10px;
    }
    .homepage-header p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    .article-links-container {
        display: flex;
        flex-wrap: wrap; /* Allows items to wrap to the next line on smaller screens */
        justify-content: center;
        padding: 30px;
        gap: 20px; /* Space between article cards */
    }
    .article-card-wrapper { /* New wrapper for card and button */
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%; /* Full width on small screens */
        max-width: 250px; /* Max width for larger screens */
    }
    .article-card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        width: 100%; /* Take full width of wrapper */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        text-align: center;
        margin-bottom: 10px; /* Space between card and button */
    }
    .article-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .article-card h3 {
        color: #007bff;
        margin-top: 0;
        font-size: 1.3em;
        margin-bottom: 15px;
    }
    /* Streamlit buttons have their own styling, but we keep these for general reference */
    .read-button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1em;
        transition: background-color 0.2s ease;
        width: 100%;
    }
    .read-button:hover {
        background-color: #0056b3;
    }
    .article-content {
        padding: 30px;
        line-height: 1.6;
        font-size: 1.05em;
    }
    .article-content h2 {
        color: #4a4a4a;
        margin-bottom: 20px;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }
    .comment-section {
        border-top: 1px solid #eee;
        margin-top: 30px;
        padding-top: 20px;
    }
    .comment-input textarea {
        width: 100%;
        padding: 12px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        border-radius: 8px;
        resize: vertical;
        font-size: 1em;
    }
    .comment-button {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1em;
        transition: background-color 0.2s ease;
        width: 100%;
    }
    .comment-button:hover {
        background-color: #218838;
    }
    .comments-display {
        background-color: #f5f5f5;
        border: 1px solid #e9e9e9;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        max-height: 300px; /* Limit height and add scroll */
        overflow-y: auto;
    }
    .comment-item {
        border-bottom: 1px dashed #ddd;
        padding: 10px 0;
        word-wrap: break-word; /* Ensure long words break */
    }
    .comment-item:last-child {
        border-bottom: none;
    }
    .comment-author {
        font-weight: bold;
        color: #555;
        margin-bottom: 5px;
        font-size: 0.9em;
    }
    .comment-text {
        font-style: italic;
        color: #666;
    }

    /* Responsive adjustments */
    @media (max-width: 600px) {
        .homepage-header {
            padding: 30px 15px;
        }
        .homepage-header h1 {
            font-size: 2em;
        }
        .homepage-header p {
            font-size: 1em;
        }
        .article-card-wrapper {
            max-width: none; /* Allow full width on small screens */
        }
        .article-links-container, .article-content {
            padding: 20px;
        }
    }

    /* Keyframe for fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
"""

# --- Article Data ---
articles_data = {
    "article1": {
        "title": "The Future of AI: Beyond Imagination",
        "content": """
        <p>Artificial intelligence continues to evolve at an astonishing pace, pushing the boundaries of what's possible. From advanced machine learning algorithms to neural networks capable of complex problem-solving, AI is poised to revolutionize industries and daily life.</p>
        <p>Experts predict that the next decade will see AI integrated into almost every aspect of society, from personalized medicine to autonomous transportation and smarter cities. Ethical considerations and responsible development will be key to harnessing its full potential.</p>
        """
    },
    "article2": {
        "title": "Climate Change: Urgent Action Required",
        "content": """
        <p>The latest scientific reports underscore the urgency of addressing climate change. Rising global temperatures, extreme weather events, and biodiversity loss are stark reminders of the challenges humanity faces.</p>
        <p>Innovation in renewable energy, sustainable agriculture, and carbon capture technologies offer hope, but systemic changes in policy and individual behavior are crucial for a resilient future. International cooperation is paramount to mitigate the impacts.</p>
        """
    },
    "article3": {
        "title": "Space Exploration: A New Golden Age",
        "content": """
        <p>We are witnessing a new golden age of space exploration, driven by both government agencies and private companies. Missions to Mars, the Moon, and beyond are pushing the frontiers of human knowledge and technological capability.</p>
        <p>From the search for extraterrestrial life to establishing permanent lunar bases, the discoveries made in space promise to transform our understanding of the universe and inspire future generations of scientists and engineers.</p>
        """
    },
    "article4": {
        "title": "Revolutionizing Healthcare with Biotechnology",
        "content": """
        <p>Biotechnology is transforming healthcare, offering groundbreaking solutions for diseases that were once incurable. Gene editing, personalized therapies, and advanced diagnostics are leading to unprecedented improvements in patient outcomes.</p>
        <p>The convergence of biology and technology is enabling new approaches to drug discovery, disease prevention, and regenerative medicine, promising a healthier future for all.</p>
        """
    },
    "article5":
    {
        "title": "The Rise of Quantum Computing",
        "content": """
        <p>Quantum computing, once a theoretical concept, is rapidly moving towards practical applications. Leveraging the principles of quantum mechanics, these powerful machines can solve problems intractable for classical computers.</p>
        <p>While still in its early stages, quantum computing holds immense potential for fields like cryptography, material science, and drug discovery, promising to unlock new frontiers of computation and innovation.</p>
        """
    }
}

# --- Streamlit Application Functions ---

def render_homepage():
    """
    Renders the homepage with a header and clickable article cards.
    Clicking a card's button updates the session state to navigate to that article.
    """
    st.markdown(f"""
        <div class="homepage-header">
            <h1>The Daily Dispatch</h1>
            <p>Your source for breaking news and in-depth analysis.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="article-links-container">', unsafe_allow_html=True)
    # Use st.columns to create a responsive grid for article cards
    # Adjust the number of columns based on screen width if needed, or use a fixed number
    num_articles = len(articles_data)
    cols = st.columns(min(num_articles, 3)) # Max 3 columns for larger screens

    for i, (article_id, data) in enumerate(articles_data.items()):
        with cols[i % len(cols)]: # Cycle through columns for layout
            st.markdown(f"""
                <div class="article-card-wrapper">
                    <div class="article-card">
                        <h3>{data["title"]}</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Read Article", key=f"read_button_{article_id}"):
                st.session_state.current_page = article_id
                st.experimental_rerun() # Rerun to switch to the article page
    st.markdown('</div>', unsafe_allow_html=True)


def render_article_page(article_id, title, content):
    """
    Renders a single article page, including its content and a comment section.
    Comments are stored in session state and are not persistent across app restarts.
    """
    st.markdown(f"""
        <div class="article-content">
            <h2>{title}</h2>
            {content}
            <div class="comment-section">
                <h3>Comments</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display existing comments
    st.markdown('<div class="comments-display">', unsafe_allow_html=True)
    if st.session_state.comments[article_id]:
        # Display comments in reverse order (newest first)
        for comment_data in reversed(st.session_state.comments[article_id]):
            st.markdown(f"""
                <div class="comment-item">
                    <div class="comment-author">Anonymous ({comment_data['timestamp']}):</div>
                    <div class="comment-text">{comment_data['text']}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p>No comments yet. Be the first to comment!</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Comment input and submit button
    # Using a unique key for the text_area allows it to be cleared programmatically
    new_comment_key = f"comment_input_{article_id}"
    new_comment = st.text_area(
        "Write your comment here...",
        key=new_comment_key,
        height=100
    )

    if st.button("Post Comment", key=f"post_comment_button_{article_id}"):
        if new_comment.strip():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.comments[article_id].append({
                'text': new_comment.strip(),
                'timestamp': current_time
            })
            st.success("Comment posted!")
            # Clear the text area by resetting its value in session state
            st.session_state[new_comment_key] = ""
            st.experimental_rerun() # Rerun to display new comment and clear input
        else:
            st.warning("Comment cannot be empty.")

    # Back to Home button
    st.markdown("---")
    if st.button("← Back to Home", key=f"back_to_home_{article_id}"):
        st.session_state.current_page = "home"
        st.experimental_rerun() # Rerun to switch to the homepage


def main():
    st.set_page_config(layout="centered", page_title="The Daily Dispatch")

    # Embed the custom CSS at the very top
    st.markdown(CSS_STYLE, unsafe_allow_html=True)

    # Initialize session state variables
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home" # Default to homepage

    if 'comments' not in st.session_state:
        st.session_state.comments = {f"article{i}": [] for i in range(1, len(articles_data) + 1)}


    # Main content rendering based on current_page
    if st.session_state.current_page == "home":
        render_homepage()
    else:
        # Check if the current_page is a valid article ID
        if st.session_state.current_page in articles_data:
            article_id = st.session_state.current_page
            article_info = articles_data[article_id]
            render_article_page(article_id, article_info["title"], article_info["content"])
        else:
            # Fallback if an invalid page is somehow set
            st.error("Page not found. Returning to homepage.")
            st.session_state.current_page = "home"
            st.experimental_rerun()


if __name__ == "__main__":
    main()
