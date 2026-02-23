import base64
import os

from pgx.yahtzee import State as YahtzeeState

def _make_yahtzee_dwg(dwg, state: YahtzeeState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    PADDING = 0.5 * GRID_SIZE
    color_set = config["COLOR_SET"]
    
    board_g = dwg.g(id="board")

    PATH = {
        "card": "images/yahtzee/scorecard.svg",
    }

    board_y_offset = GRID_SIZE
    # --- Background ---
    with open(
        os.path.join(os.path.dirname(__file__), PATH["card"]),
        "rb"
    ) as f:
        b64_img = base64.b64encode(f.read())

    board_g.add(
        dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=(- GRID_SIZE * 4.3 / 2,board_y_offset),
            size=(GRID_SIZE * 10, GRID_SIZE * 10)
        )
    )

    # Players
    player_section = board_g.add(dwg.g(id="players", fill=color_set.grid_color))
    start_x = GRID_SIZE * 2.15
    start_y = GRID_SIZE * 3.1
    spacing_x = GRID_SIZE * 0.9
    for i in range(state._num_players):
        player_section.add(
            dwg.text(
                text=f"Player {i}",
                insert=(start_x + i * spacing_x, start_y),
                font_size="9px",
                font_family="Arial",
                font_weight= "bold" if i == state.current_player else "normal",
            )
        )
    
    # Scores
    upper_score_section = board_g.add(dwg.g(id="upper-score", fill=color_set.grid_color))
    lower_score_section = board_g.add(dwg.g(id="lower-score", fill=color_set.grid_color))
    spacing_y = GRID_SIZE * 0.39
    for j in range(state._num_players):
        _start_x = start_x + j * spacing_x
        
        for i in range(6):
            _start_y = start_y + GRID_SIZE * 0.34 + spacing_y * i
            if state._categories_used[j, i] == True:
                upper_score_section.add(
                    dwg.text(
                        text=f"{state._scores[j, i]}",
                        insert=(_start_x, _start_y),
                        font_size="9px",
                    )
                )
        for i in range(7):
            _start_y = start_y + GRID_SIZE * 0.35 + spacing_y * 9.8 +  + i * spacing_y

            if state._categories_used[j, i + 6] == True:
                lower_score_section.add(
                    dwg.text(
                        text=f"{state._scores[j, i + 6]}",
                        insert=(_start_x, _start_y),
                        font_size="9px",
                    )
                )
    
    # dice
    dice_section = board_g.add(dwg.g(id="dice", fill=color_set.grid_color))

    def _get_dice_g(face:int, insert):
        file_path = f"images/yahtzee/dice_{str(face)}.svg"
        with open(os.path.join(os.path.dirname(__file__), file_path), "rb") as f:
            b64_image = base64.b64encode(f.read())
        
        return dwg.image(
            "data:image/svg+xml;base64," + b64_image.decode("ascii"),
            insert=insert,
            size=(GRID_SIZE * 0.75, GRID_SIZE * 0.75)
        )
    
    dice_x_spacing = GRID_SIZE
    for i in range(5):
        dice_section.add(
            _get_dice_g(
                face=state._dice[i],
                insert=(i * dice_x_spacing, 0)
            )
        )
    dice_section.add(
        dwg.text(
            text=f"{state._rolls_left} rolls left",
            insert=(5 * dice_x_spacing, GRID_SIZE * 0.4)
        )
    )


    return board_g
