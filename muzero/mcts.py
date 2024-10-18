import math
import random
from typing import Dict, List, Tuple

import torch
from torchtyping import TensorType

def run_mcts(root: 'Node', network: 'MuZeroNetwork', config: 'MuZeroConfig') -> None:
    """
    Perform Monte Carlo Tree Search (MCTS) simulations to expand and evaluate the search tree.
    
    Each simulation consists of the following phases:
        1. Selection: Traverse the tree using the UCB1 formula to select promising nodes.
        2. Expansion: Expand leaf nodes by adding all possible child actions.
        3. Evaluation: Use the neural network to evaluate the leaf node.
        4. Backup: Propagate the evaluation results up the tree.
    
    Mathematical Foundations:
        - UCB1 Formula: Balances exploration and exploitation during action selection.
            UCB(s_k, a) = Q(s_k, a) + c_{puct} * P(s_k, a) * (sqrt(Σ_b N(s_k, b)) / (1 + N(s_k, a)))
        - Bellman Backup: Updates value estimates using the Bellman equation.
            v = r + γ * v'
        - Softmax Normalization: Converts logits to probability distributions.
            π̂_k(a) = softmax(π̂_k)_a = exp(π̂_k(a)) / Σ_{b} exp(π̂_k(b))
    
    References:
        - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" by David Silver et al.
        - Upper Confidence Bound formulas in reinforcement learning literature.
    
    Parameters:
        root (Node): The root node of the current search tree representing state s₀.
        network (MuZeroNetwork): The neural network providing h_θ, g_θ, f_θ.
        config (MuZeroConfig): Configuration parameters including hyperparameters.
    """
    for simulation in range(config.num_simulations):
        node = root
        search_path = [node]
        value = 0
        
        # ====================
        # SELECTION Phase
        # ====================
        while node.is_expanded():
            # Compute UCB values for all child actions:
            # UCB(s_k, a) = Q(s_k, a) + U(s_k, a)
            ucb_values = {
                action: node.child_Q(action) + node.child_U(action, config)
                for action in node.children
            }
            # Select the action a* with the highest UCB value.
            best_action = argmax(ucb_values)
            node = node.children[best_action]
            search_path.append(node)
        
        # ====================
        # EXPANSION Phase
        # ====================
        # Retrieve state s_k and action a_k leading to this node.
        s_k = node.state
        a_k = node.parent_action
        
        if node.parent is None:
            # Root node: Apply the representation function h_θ to o₀ to obtain s₀.
            # s₀ = h_θ(o₀)
            s_k = network.representation(s_k.observation)
        else:
            # Non-root node: Apply the dynamics function g_θ to (s_{k-1}, a_{k-1}) to obtain s_k and r_k.
            # s_k, r_k = g_θ(s_{k-1}, a_{k-1})
            s_k, r_k = network.dynamics(node.parent.hidden_state, a_k)
            node.reward = r_k  # Immediate reward from transition.
        
        # Update the node's hidden state with s_k.
        node.hidden_state = s_k
        
        # ====================
        # EVALUATION Phase
        # ====================
        # Use the prediction function f_θ to obtain policy logits π̂_k and value v̂_k.
        # (π̂_k, v̂_k) = f_θ(s_k)
        pi_k, v_k = network.prediction(s_k)
        
        # Expand the node by adding child nodes for each possible action with prior probabilities.
        # This corresponds to expanding the state s_k by all possible actions a ∈ A_k.
        node.expand(pi_k)
        
        # ====================
        # BACKUP Phase
        # ====================
        # Backpropagate the value estimate v_k up the search path using the Bellman equation.
        # For each node in the search path, update N(s_k, a_k), W(s_k, a_k), and Q(s_k, a_k).
        backpropagate(search_path, v_k, config)

    # Collect child visit counts
    child_visits = {action: child.visit_count for action, child in root.children.items()}
    return {'child_visits': child_visits}


def backpropagate(search_path: List['Node'], value: float, config: 'MuZeroConfig') -> None:
    """
    Backpropagate the value estimate up the search path, updating visit counts and value sums.
    
    For each node in the reversed search path, perform the following updates:
        N(s_k, a_k) ← N(s_k, a_k) + 1
        W(s_k, a_k) ← W(s_k, a_k) + v
        V(s_k, a_k) ← W(s_k, a_k) / N(s_k, a_k)
        v ← r_k + γ * v  # Update value for the next node up.
    
    Mathematical Concepts:
        - Visit Count Update: N(s_k, a_k) += 1
        - Value Sum Update: W(s_k, a_k) += v
        - Mean Value Calculation: V(s_k, a_k) = W(s_k, a_k) / N(s_k, a_k)
        - Recursive Value Update: v ← r_k + γ * v
    
    Mathematical Association:
        This implements the Bellman backup equation:
            V(s_k, a_k) = R(s_k, a_k) + γ V(s_{k'}, a_{k'})
        Where:
            - s_{k'} is the next state following action a_k.
            - R(s_k, a_k) is the immediate reward.
    
    Parameters:
        search_path (list of Node): The sequence of nodes traversed during selection.
        value (float): The initial value estimate v_L from the leaf node evaluation.
        config (MuZeroConfig): Configuration parameters including discount factor γ.
    """
    for node in reversed(search_path):
        node.visit_count += 1  # Update visit count: N(s_k, a_k) ← N(s_k, a_k) + 1
        node.value_sum += value  # Update total value: W(s_k, a_k) ← W(s_k, a_k) + v
        node.value = node.value_sum / node.visit_count  # Q(s_k, a_k) ← W / N
        
        # Update value with discounted future value: v ← r_k + γ * v
        value = node.reward + config.discount * value  # Bellman Backup Equation


class Node:
    def __init__(self, state: 'State'):
        """
        Initialize a tree node representing a latent state s_k.
        
        Parameters:
            state (State): The latent state associated with this node.
        
        Mathematical Representation:
            s_k ∈ ℝ^{|h|}
        """
        self.state: 'State' = state  # Latent state s_k ∈ ℝ^{|h|}
        self.hidden_state: TensorType["hidden_dim"] | None = None  # Hidden state representation h_θ(s_k)
        self.reward: float = 0  # Immediate reward r_k obtained from transition to s_k.
        self.visit_count: int = 0  # Visit count N(s_k, a_k)
        self.value_sum: float = 0  # Total value W(s_k, a_k)
        self.children: Dict[int, 'Node'] = {}  # Child nodes for each action a ∈ A
        self.parent: 'Node' | None = None  # Parent node in the search tree
        self.parent_action: int | None = None  # Action a_k taken to reach this node
        self.prior: float = 0  # Prior probability P(s_k, a_k) from the policy π̂_k
    
    def is_expanded(self):
        """
        Determine whether the node has been expanded (i.e., has child nodes).
        
        Returns:
            bool: True if expanded, False otherwise.
        
        Mathematical Interpretation:
            A node is expanded if ∃ a ∈ A such that child(a) is present.
        """
        return len(self.children) > 0
    
    def expand(self, pi: TensorType["action_space"]):
        """
        Expand the current node by adding child nodes for each possible action.
        
        Parameters:
            pi (torch.Tensor): Policy logits π̂_k for all actions a ∈ A.
        
        Mathematical Process:
            - Apply softmax to π̂_k to obtain P(s_k, a_k) for each action.
            - Instantiate child nodes s_{k+1} = g_θ(s_k, a_k) for each a_k ∈ A.
            - Assign prior probabilities P(s_k, a_k) to each child node.
        
        References:
            - Section 3.3: Policy and Value Functions
        """
        # Apply softmax to obtain prior probabilities P(s_k, a_k).
        policy = torch.softmax(pi, dim=-1).detach().cpu().numpy()
        
        for a, p in enumerate(policy):
            # Initialize child nodes with the next state s_{k+1} = g_θ(s_k, a_k).
            child_state = self.state.next_state(a)
            child = Node(child_state)
            child.parent = self
            child.parent_action = a
            child.prior = p  # P(s_k, a_k) from the policy network.
            self.children[a] = child
    
    def child_Q(self, a: int) -> float:
        """
        Compute the mean action-value Q(s_k, a_k) for a given action a.
        
        Parameters:
            a (int): Action index.
        
        Returns:
            float: Mean action-value Q(s_k, a_k).
        
        Mathematical Definition:
            Q(s_k, a_k) = W(s_k, a_k) / N(s_k, a_k)
            Where W(s_k, a_k) is the total value sum, and N(s_k, a_k) is the visit count.
        
        Reference:
            - Equation (4) in the MuZero paper
        """
        if self.children[a].visit_count == 0:
            return 0
        return self.children[a].value_sum / self.children[a].visit_count  # Q(s_k, a_k) = W / N
    
    def child_U(self, a: int, config: 'MuZeroConfig') -> float:
        """
        Compute the exploration term U(s_k, a_k) for the Upper Confidence Bound formula.
        
        Parameters:
            a (int): Action index.
            config (MuZeroConfig): Configuration parameters including c_{puct} and γ.
        
        Returns:
            float: Exploration term U(s_k, a_k).
        
        Mathematical Definition:
            U(s_k, a_k) = c_{puct} * P(s_k, a_k) * (sqrt(N(s_k, ·)) / (1 + N(s_k, a_k)))
            Where:
                - c_{puct} is the exploration constant.
                - P(s_k, a_k) is the prior probability of action a_k.
                - N(s_k, ·) is the total visit count for all actions at s_k.
                - N(s_k, a_k) is the visit count for action a_k.
        
        Reference:
            - Equation (2) in the MuZero paper
        """
        total_visits = sum(child.visit_count for child in self.children.values())  # Σ_b N(s_k, b)
        if self.visit_count == 0:
            return 0
        return (
            config.c_puct * self.children[a].prior *
            (math.sqrt(total_visits) / (1 + self.children[a].visit_count))
        )
    
    def value(self) -> float:
        """
        Retrieve the mean action-value Q(s_k, a_k) for this node.
        
        Returns:
            float: Mean action-value Q(s_k, a_k).
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count  # Q(s_k, a_k) = W / N


def argmax(dictionary: Dict[int, float]) -> int:
    """
    Return the key with the highest value in the dictionary.
    
    Parameters:
        dictionary (dict): A dictionary mapping keys to numerical values.
    
    Returns:
        key: The key corresponding to the highest value.
    
    Mathematical Concept:
        argmax_{a ∈ A} f(a)
    """
    return max(dictionary, key=dictionary.get)
