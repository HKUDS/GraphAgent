few_shot_example_1 = \
(
    """
    Input:
    Knowledge Text:
    Paper Title: Simple Neural Network for Image Classification
    Abstract: We present a basic neural network architecture for image classification. Our model uses three layers and achieves 85% accuracy on the test dataset. The implementation is straightforward and suitable for educational purposes. Experimental results show that this simple architecture can serve as a good baseline for more complex models.

    User Annotation:
    Which category should the paper be classified into? You have the following choices: Distributed Parallel and Cluster Computing (cs.DC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE)
    """.strip()
    ,
    """
    Output:
    {{
        "thinking": "1. Understanding the task: This is a paper classification task where we need to categorize a research paper into different CS fields
                 2. Analyzing the input: We have a paper about neural networks and image classification, with both title and abstract available
                 3. For paper classification tasks, we typically need to understand:
                    - What the paper is about (paper identity)
                    - What background/context it builds upon
                    - What research questions/problems it addresses
                    - How it solves the problem
                    - What it achieves/proves
                 4. These aspects help determine which CS field it belongs to",
        "scaffold_nodes": [{{"id": 0, "type": "paper", "name": "Simple Neural Network for Image Classification"}}, {{id: 1, "type": "research_background", "name": "research_background"}}, {{id: 2, "type": "research_question", "name": "research_question"}}, {{"id": 3, "type": "methodology", "name": "methodology"}}, {{id: 4, "type": "key_results", "name": "key_results"}}],
    }}
    """.strip()
)

few_shot_example_2 = \
(
"""
Input:
Knowledge Text:
Story Title: Romeo and Juliet (Act 1 Scene 1)
Text: Two households, both alike in dignity, In fair Verona, where we lay our scene, From ancient grudge break to new mutiny, Where civil blood makes civil hands unclean. From forth the fatal loins of these two foes, A pair of star-cross'd lovers take their life. The fearful passage of their death-mark'd love, And the continuance of their parents' rage, Which, but their children's end, nought could remove.
User Annotation:
Analyze the key themes, characters, and relationships in this story. Focus on how family dynamics and social conflicts influence the plot development.
""".strip()
,
"""
Output:
{{
"thinking": "1. Understanding the task: This is a literary analysis task focusing on themes, characters, and relationships
2. Analyzing the input: We have a complex narrative involving family conflicts, romance, and tragedy
3. For story analysis with focus on relationships and conflicts, we need to understand:
- Who are the key players and their relationships
- What are the social and familial contexts
- What conflicts drive the narrative
- How these elements interact over time
- What themes emerge from these interactions
4. These aspects help analyze how family dynamics influence the story",
"scaffold_nodes": [{{"id": 0, "type": "family_groups", "name": "opposing_households"}}, {{"id": 1, "type": "social_context", "name": "verona_society"}}, {{"id": 2, "type": "central_conflict", "name": "family_feud"}}, {{"id": 3, "type": "character_relationships", "name": "star_crossed_lovers"}}, {{"id": 4, "type": "thematic_elements", "name": "tragedy_themes"}}]
}}
""".strip()
)

few_shot_example_3 = \
(
"""
Input:
Knowledge Text:
Report Title: Global Climate Summit 2024 Outcomes
Text: The 2024 Global Climate Summit concluded with major nations announcing ambitious carbon reduction targets. China pledged 40% reduction by 2030, while the EU increased its target to 55%. The US rejoined previous commitments and added new renewable energy initiatives. Developing nations received promises of $100B in annual climate funding. However, environmental groups criticized the lack of immediate action plans and binding agreements.
User Annotation:
Summarize the key decisions, stakeholder positions, and potential impacts from this climate summit report.
""".strip()
,
"""
Output:
{{
"thinking": "1. Understanding the task: This is a news analysis task focusing on policy decisions and their implications
2. Analyzing the input: We have a complex policy report involving multiple stakeholders, commitments, and reactions
3. For policy report analysis, we need to understand:
- What major decisions were made
- Who the key stakeholders are
- What commitments were promised
- What criticisms or challenges exist
- What potential impacts might occur
4. These aspects help analyze the comprehensive impact of the summit",
"scaffold_nodes": [{{"id": 0, "type": "policy_decisions", "name": "carbon_reduction_targets"}}, {{"id": 1, "type": "stakeholders", "name": "participating_nations"}}, {{"id": 2, "type": "commitments", "name": "financial_pledges"}}, {{"id": 3, "type": "challenges", "name": "implementation_concerns"}}, {{"id": 4, "type": "impact_assessment", "name": "potential_outcomes"}}]
}}
""".strip()
)