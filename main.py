import cv2
import numpy as np
from PIL import Image
import os
import pyautogui
import mss
import time

paused = False

# Константы
GRID_SIZE = 8  # Размер сетки

def preprocess_image(image):
    return image  # Возвращаем цветное изображение

def resize_icon(icon, size):
    width, height = icon.size
    aspect_ratio = width / height
    new_width = size
    new_height = int(size / aspect_ratio)
    return icon.resize((new_width, new_height), Image.LANCZOS)

def load_icon_templates():
    icon_templates = {
        'brood': ['brood.jpg', 'brood_4.png', 'brood_5.png'],
        'venge': ['venge.jpg', 'venge_4.png', 'venge_5.png'],
        'cm': ['cm.jpg', 'cm_4.png', 'cm_5.png'],
        'lich': ['lich.jpg', 'lich_4.png', 'lich_5.png'],
        'lina': ['lina.jpg', 'lina_4.png', 'lina_5.png'],
        'wyvern': ['wyvern.jpg', 'wyvern_4.png', 'wyvern_5.png']
    }
    loaded_templates = {}
    screen_width, screen_height = pyautogui.size()
    grid_size = int(screen_height * (120 / 1440)) 

    for name, files in icon_templates.items():
        loaded_templates[name] = []
        for file in files:
            if os.path.exists(file):
                template = Image.open(file)
                resized_template = resize_icon(template, grid_size)  # Размер шаблона 120x120 пикселей
                img_np = np.array(resized_template)
                loaded_templates[name].append((cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), file))
            else:
                print(f"Template file {file} not found, skipping.")
    return loaded_templates


def detect_icons(image, templates):
    icons = []
    screen_width, screen_height = pyautogui.size()
    grid_size = int(screen_height * (120 / 1440))  # Размер ячейки 120x120 пикселей
    print(grid_size)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Извлекаем квадрат 120x120 пикселей из изображения
            x1, y1 = col * grid_size, row * grid_size
            x2, y2 = x1 + grid_size, y1 + grid_size
            cell = image[y1:y2, x1:x2]

            best_match = None
            best_match_score = -1
            for name, files in templates.items():
                for item in files:
                    if item is None:
                        continue
                    template, filename = item
                    res = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_match_score:
                        best_match_score = max_val
                        best_match = (x1 + grid_size // 2, y1 + grid_size // 2, name, filename)

            if best_match_score >= 0.6 and best_match is not None:  # Порог для совпадения
                icons.append(best_match)

    return icons


def create_grid(icons, image):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    for (x, y, name, filename) in icons:
        # Преобразуем координаты в индексы сетки
        grid_x = int(x / (image.shape[1] / GRID_SIZE))
        grid_y = int(y / (image.shape[0] / GRID_SIZE))
        # Используем индекс имени иконки в словаре templates
        grid[grid_y, grid_x] = list(templates.keys()).index(name) + 1  # Индекс +1, чтобы избежать нулевых значений и сделать их положительными
    
    return grid

def draw_icons_on_image(image, icons, valid_moves):
    output_image = image.copy()

    # Отрисовка иконок
    for (x, y, name, filename) in icons:
        cv2.circle(output_image, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.putText(output_image, name, (int(x) - 20, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Отрисовка стрелочек для допустимых движений
    for move in valid_moves:
        (r1, c1), (r2, c2), combinations = move
        pt1 = (int(c1 * (image.shape[1] / GRID_SIZE) + (image.shape[1] / GRID_SIZE) // 2),
               int(r1 * (image.shape[0] / GRID_SIZE) + (image.shape[0] / GRID_SIZE) // 2))
        pt2 = (int(c2 * (image.shape[1] / GRID_SIZE) + (image.shape[1] / GRID_SIZE) // 2),
               int(r2 * (image.shape[0] / GRID_SIZE) + (image.shape[0] / GRID_SIZE) // 2))
        
        # Определение цвета и размера стрелочки в зависимости от типа комбинации
        color = (0, 255, 0)  # Зеленый для 3 элементов
        thickness = 2
        if any(combo[-1] == 'five' for combo in combinations):
            color = (0, 0, 255)  # Красный для 5 элементов
            thickness = 4
        elif any(combo[-1] == 'four' for combo in combinations):
            color = (0, 0, 255)  # Красный для 4 элементов
            thickness = 3

        # Отрисовка стрелочки
        cv2.arrowedLine(output_image, pt1, pt2, color, thickness)

    return output_image

def find_valid_moves(board):
    valid_moves = []
    rows, cols = board.shape

    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:  # Swap with right
                new_board = swap_elements(board, (r, c), (r, c + 1))
                combinations = check_combinations(new_board)
                if combinations:
                    valid_moves.append(((r, c), (r, c + 1), combinations))

            if r < rows - 1:  # Swap with below
                new_board = swap_elements(board, (r, c), (r + 1, c))
                combinations = check_combinations(new_board)
                if combinations:
                    valid_moves.append(((r, c), (r + 1, c), combinations))

    # Сортировка движений по приоритету комбинаций
    def priority(combo_type):
        if combo_type == 'five':
            return 3
        elif combo_type == 'four':
            return 2
        elif combo_type == 'three':
            return 1
        return 0

    valid_moves.sort(key=lambda move: max(priority(combo[-1]) for combo in move[2]), reverse=True)

    return valid_moves

def swap_elements(board, pos1, pos2):
    board = board.copy()
    board[pos1], board[pos2] = board[pos2], board[pos1]
    return board

def check_combinations(board):
    rows, cols = board.shape
    moves = []

    def check_line(line, r=None, c=None):
        combos = []
        combo_start = 0
        for i in range(1, len(line)):
            if line[i] == line[i - 1] and line[i] != 0:
                length = i - combo_start + 1
                if length >= 5:
                    combos.append((combo_start, i, length, 'five'))
                elif length == 4:
                    combos.append((combo_start, i, length, 'four'))
                elif length == 3:
                    if i + 1 < len(line) and line[i + 1] == line[i]:
                        combos.append((combo_start, i + 1, length + 1, 'three'))
                    else:
                        combos.append((combo_start, i, length, 'three'))
            else:
                combo_start = i
        return combos

    # Проверка горизонтальных и вертикальных линий
    for r in range(rows):
        line_combos = check_line(board[r, :], r=r)
        for start, end, length, combo_type in line_combos:
            moves.append(('row', r, start, end, combo_type))

    for c in range(cols):
        line_combos = check_line(board[:, c], c=c)
        for start, end, length, combo_type in line_combos:
            moves.append(('col', c, start, end, combo_type))

    # Check for cross-patterns
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = board[r, c]
            if center == 0:
                continue
            if (board[r, c - 1] == center and board[r, c + 1] == center and 
                board[r - 1, c] == center and board[r + 1, c] == center):
                moves.append(('cross', r, c, 'five'))

    return moves

def perform_move(move, region):
    (r1, c1), (r2, c2), _ = move

    # Размеры региона захвата
    top = region["top"]
    left = region["left"]
    cell_width = image.shape[1] / GRID_SIZE
    cell_height = image.shape[0] / GRID_SIZE

    # Преобразуем координаты в пиксели относительно всего экрана
    x1 = int(left + c1 * cell_width + cell_width // 2)
    y1 = int(top + r1 * cell_height + cell_height // 2)
    x2 = int(left + c2 * cell_width + cell_width // 2)
    y2 = int(top + r2 * cell_height + cell_height // 2)

    # Выполнение движения мышью
    pyautogui.moveTo(x1, y1)
    pyautogui.mouseDown();
    pyautogui.moveTo(x2,y2, 0.11, pyautogui.easeInQuad)
    pyautogui.mouseUp()
    
def log_moves(valid_moves):
    if not valid_moves:
        print("No valid moves found.")
        return

    print("Possible Moves:")
    for (start, end, combinations) in valid_moves:
        (x1, y1) = start
        (x2, y2) = end
        print(f"Move from ({x1}, {y1}) to ({x2}, {y2})")
        for combo_type in combinations:
            print(f"  Combination type: {combo_type}")

    # Выводим только первое движение с наивысшим приоритетом
    (start, end, combinations) = valid_moves[0]
    (x1, y1) = start
    (x2, y2) = end
    print(f"\nSelected Move with highest priority:")
    print(f"Move from ({x1}, {y1}) to ({x2}, {y2})")
    for combo_type in combinations:
        print(f"  Combination type: {combo_type}")

def load_image_from_file(filepath=None):
    if filepath:
        image = Image.open(filepath)
        img_np = np.array(image)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        # Если filepath не задан, используем захват экрана
        region = get_dynamic_region()
        return capture_screen(region=region)

def capture_screen(region=None):
    with mss.mss() as sct:
        if region:
            screen = sct.grab(region)
            screen = Image.frombytes('RGB', screen.size, screen.bgra, 'raw', 'BGRX')
        else:
            screen = sct.grab(sct.monitors[1])  # Захват экрана всего монитора
        img_np = np.array(screen)
    return cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

def save_image(image, filepath):
    cv2.imwrite(filepath, image)

def get_dynamic_region():
    screen_width, screen_height = pyautogui.size()
    # Используем процентное соотношение для адаптации к экрану
    top = int(screen_height * (180 / 1440))  # Пропорция 150 пикселей относительно разрешения 1440p
    left = int(screen_width * (300 / 2560))  # Пропорция 275 пикселей относительно разрешения 2560p
    width = int(screen_width * (960 / 2560)) # Пропорция 1000 пикселей относительно разрешения 2560p
    height = int(screen_height * (960 / 1440)) # Пропорция 1000 пикселей относительно разрешения 1440p
    
    return {"top": top, "left": left, "width": width, "height": height}
def main():
    global image
    global templates
    
    # Загрузка шаблонов иконок
    templates = load_icon_templates()
    while True:
        image = load_image_from_file()  # Захват экрана по умолчанию
        processed_image = preprocess_image(image)
        save_image(processed_image, 'result_with_moves.png')
        
        # Детектирование иконок
        icons = detect_icons(processed_image, templates)
        print(f"Found {len(icons)} icons")
    
        # Создание сетки
        grid = create_grid(icons, image)
        print(grid)
    
        # Поиск допустимых ходов
        valid_moves = find_valid_moves(grid)
    
        if valid_moves:  # Проверка, что valid_moves не пустой
            best_move = None
            for move in reversed(valid_moves):
                (start, end, combinations) = move
                
                # Поиск движения с наивысшим приоритетом
                has_five = any(combo[-1] == 'five' for combo in combinations)
                has_four = any(combo[-1] == 'four' for combo in combinations)
                has_three = any(combo[-1] == 'three' for combo in combinations)

                if has_five:
                    best_move = move
                    break  # Если нашли 'five', то это наивысший приоритет
                elif has_four:
                    best_move = move
                elif has_three:
                    if best_move is None:  # Если еще не выбрано лучшее движение
                        best_move = move

            print(best_move)
            if best_move and not paused:
                region = get_dynamic_region()
                perform_move(best_move, region)
        
        time.sleep(0.5)  # Ждем 0.50 секунды перед следующим движением

if __name__ == "__main__":
    main()


