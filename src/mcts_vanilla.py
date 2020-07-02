
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.

def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.

    Returns:        A node from which the next stage of the search can proceed.

    """
    
    if(len(node.untried_actions)>0):
        leaf_node = node   
        return (leaf_node, state)
    else:
        children = []
        myMove=(identity == board.current_player(state))
        for child in node.child_nodes:
            value=UCT(node, node.child_nodes[child],myMove)
            children.append((value,child))
            
        children.sort(key=byVal,reverse=True)
             
        for child in children:
            possible_leaf = node.child_nodes[child[1]]
            possible_leaf_pair = traverse_nodes(possible_leaf, board, board.next_state(state, possible_leaf.parent_action), identity)
            if(possible_leaf != None):
                return possible_leaf_pair
            
    return None
    
def byVal(e):
    return e[0]

def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    
    #get random action
    action=choice(node.untried_actions)
    (node.untried_actions).remove(action)
    new_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(board.next_state(state, action)))
    node.child_nodes[action]=new_node
    return (new_node, board.next_state(state, action))


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """
    if(board.is_ended(state)):
        return board.points_values(state)
    else:    
        action=choice(board.legal_actions(state))
        return rollout(board, board.next_state(state, action)) 


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """

    node.wins=node.wins+won
    node.visits = node.visits+1
    if(node.parent == None):
        return node
    else:
        backpropagate(node.parent, won)
        


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))
    
    fullyExplored= False
    
    for step in range(num_nodes):
        # Copy the game for sampling a playthrough
        sampled_game = state

        # Start at root
        node = root_node

        # Do MCTS - This is all you!
        pair = traverse_nodes(node, board, sampled_game, identity_of_bot)
        if(pair == None):
            fullyExplored=True
            break
        new_pair = expand_leaf(pair[0], board, pair[1])
        points = rollout(board, board.next_state(new_pair[1], new_pair[0].parent_action))
        if(points[identity_of_bot]==1):
            won=1
        else:
            won=0
        backpropagate(new_pair[0],won)
        
        

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best =[]
    for child in root_node.child_nodes:
        chi=root_node.child_nodes[child]
        if(fullyExplored):
            best.append((chi.wins/chi.visits,chi))
        else:
            best.append((chi.visits,chi))
    best.sort(key=byVal,reverse=True)
    return best[0][1].parent_action
    
    
def UCT(node, child, myMove):
    winRate = child.wins/child.visits
    if(myMove==False):
        winRate=1-winRate
    exploration = log(node.visits)/child.visits
    return winRate+explore_faction*sqrt(exploration)