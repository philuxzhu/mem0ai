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

EXTRACT_NODES_PROMPT = f"""
You are a smart assistant who understands entities and their types in a given text. Extract all the entities from the text.

## Rules:
1. The extracted entities and their types include:
   - User: The person in the current chat session.
   - Preference: The user's expressed like, dislike, or preference for something.
   - Location: The physical or virtual place where an activity occurs or an entity exists.
   - Event: A time-bound activity, occurrence, or experience.
   - Topic: The subject of conversation, interest, or field of knowledge.
   - Object: A tangible item, tool, device, or property.
   - Organization: A company, institution, group, or formal entity.
   - Document: Information content in various forms.
2. If the entity is a date (such as "tomorrow", "Tuesday" etc.), convert it to a date in the yyyy-mm-dd format. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
3. ***DO NOT*** answer the question itself if the given text is a question.
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
    - The extracted entities and their types include:
     （1）User: The person in the current chat session.
     （2）Preference: The user's expressed like, dislike, or preference for something.
     （3）Location: The physical or virtual place where an activity occurs or an entity exists.
     （4）Event: A time-bound activity, occurrence, or experience.
     （5）Topic: The subject of conversation, interest, or field of knowledge.
     （6）Object: A tangible item, tool, device, or property.
     （7）Organization: A company, institution, group, or formal entity.
     （8）Document: Information content in various forms.
    - If the entity is a date (such as "tomorrow", "Tuesday" etc.), convert it to a date in the yyyy-mm-dd format. Today's date is {datetime.now().strftime("%Y-%m-%d")}.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction.

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
