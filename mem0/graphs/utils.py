from datetime import datetime


UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Memory Format:
source -- RELATIONSHIP -- destination

Task Details:
======= Existing Graph Memories:=======
{existing_memories}

======= New Graph Memory:=======
{new_memories}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

FACT_RETRIEVAL_PROMPT = f"""You are an expert in user portrait, particularly skilled in extracting user portrait information from chat content. Your task is to extract user portrait information from the chat content.

# The types of user portrait information to be extracted include:
1. Basic information, such as name, gender, birth date, zodiac sign, occupation.
2. Interests and hobbies, the user's likes and dislikes, especially in categories such as food, products, entertainment, and sports.
3. Social relationships, such as friendships, families, schoolmates, colleagues.
4. Important milestones, progress, and key matters in work, life, and study.


# Remember the following rules:
1. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
2. Only return the three types of user portrait information mentioned above, do not output other types of information.
3. If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
4. Create the user portraits based on the user chat content only. Do not pick anything from the system messages.
5. You should detect the language of the user input and record the user portraits in the same language.
6. Output your response strictly in the following JSON structure:
{{
    "facts": [
        {{
            "username": "",        // The username of the individual，must be exactly the same as the username in the input, even if the username in the input contains emojis.
            "fact": "",            // The extracted user portrait of the user, such as name, gender, birth date, zodiac sign, occupation, interests and hobbies, friendships, families, schoolmates, colleagues, and so on.
            "time": "",            // The time of the conversation which contains user portraits
        }},
        ...
    ]
}}

# Here are some few shot examples:

Input: 
(2025-08-01 08:33:20)John: Hi, How about today?
(2025-08-01 08:33:24)Lisa: Hi, John. I have a meeting with Peter at 3pm. We will discuss the new project.
Output: 
{{
    "facts" : []
}}

Input:
(2025-08-02 10:43:08)👁️‍🗨️John: Hi, my name is John. I am a software engineer.
(2025-08-02 10:43:15)Lisa: Nice to meet you, John. I'm Lisa. This is my friend Peter.
Output:
{{
    "facts" : [
        {{
            "username": "👁️‍🗨️John",
            "fact": "Name is John",
            "time": "2025-08-02 10:43:08"
        }},
        {{
            "username": "👁️‍🗨️John",
            "fact": "Is a Software engineer",
            "time": "2025-08-02 10:43:08"
        }},
        {{
            "username": "Lisa",
            "fact": "Name is Lisa",
            "time": "2025-08-02 10:43:15"
        }},
        {{
            "username": "Lisa",
            "fact": "have a friend Peter",
            "time": "2025-08-02 10:43:15"
        }}
    ]
}}

Input:
(2025-08-19 22:10:49)👿Hong Xiao: Hi，大家好，我是小红，1997年出生，我喜欢羽毛球，很高兴认识大家。
(2025-08-19 22:11:21)😄Ming Xiao: 我也喜欢羽毛球，技术还可以。另外我最喜欢的电影是《盗梦空间》和《星际穿越》。
(2025-08-19 22:14:09)👿Hong Xiao: 我最喜欢《霸王别姬》
(2025-08-19 22:20:19)Qiang Xiao: 我喜欢打篮球，最喜欢NBA湖人队。另外今天是我生日，哈哈。
Output:
{{
    "facts" : [
        {{
            "username": "👿Hong Xiao",
            "fact": "名字是小红",
            "time": "2025-08-19 22:10:49"
        }},
        {{
            "username": "👿Hong Xiao",
            "fact": "1997年出生",
            "time": "2025-08-19 22:10:49"
        }},
        {{
            "username": "👿Hong Xiao",
            "fact": "喜欢羽毛球",
            "time": "2025-08-19 22:10:49"
        }},
        {{
            "username": "😄Ming Xiao",
            "fact": "也喜欢羽毛球",
            "time": "2025-08-19 22:11:21"
        }},
        {{
            "username": "😄Ming Xiao",
            "fact": "羽毛球技术还可以",
            "time": "2025-08-19 22:11:21"
        }},
        {{
            "username": "😄Ming Xiao",
            "fact": "最喜欢的电影是《盗梦空间》和《星际穿越》",
            "time": "2025-08-19 22:11:21"
        }},
        {{
            "username": "👿Hong Xiao",
            "fact": "最喜欢《霸王别姬》",
            "time": "2025-08-19 22:14:09"
        }},
        {{
            "username": "Qiang Xiao",
            "fact": "喜欢打篮球，最喜欢NBA湖人队",
            "time": "2025-08-19 22:20:19"
        }},
        {{
            "username": "Qiang Xiao",
            "fact": "生日是{datetime.now().strftime("%m-%d")}",
            "time": "2025-08-19 22:20:19"
        }}
    ]
}}
"""

EXTRACT_NODES_PROMPT = f"""
You are a smart assistant who understands entities and their types in a given text. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.
"""

EXTRACT_RELATIONS_PROMPT = f"""

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. The format of the messages is as follows: (message Time)sender's name: message content.
CUSTOM_PROMPT

Relationships:
    - Represents the relationship between entities, for example, "loves" in "Kendra loves Adidas shoes".
    - Use consistent, general, and timeless relationship types.
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.
    - The extracted entities should be as concise as possible, within 5 characters (in Chinese) or 5 words (in English).
    - If the entity is a date (such as "tomorrow", "Tuesday" etc.), convert it to a date in the yyyy-mm-dd format. Today's date is {datetime.now().strftime("%Y-%m-%d")}.

Relationship time:
    - The time of relationship from message.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph ext raction.

You should detect the language of the user input and make sure the extracted source、relationship and destination be in the same language."""

DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes. 

For example: 
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted.
"""


def get_delete_messages(existing_memories_string, data, user_id):
    return DELETE_RELATIONS_SYSTEM_PROMPT.replace(
        "USER_ID", user_id
    ), f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}"
