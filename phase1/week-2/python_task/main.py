from dataclasses import dataclass

# ========= task 1 classes =========


@dataclass
class Point:
    x: int | float
    y: int | float

    def __str__(self):
        return f"({self.x}, {self.y})"


@dataclass
class Line:
    p1: Point
    p2: Point

    @property
    def length(self):
        return ((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2) ** 0.5


@dataclass
class Circle:
    center: Point
    radius: int | float

    @property
    def area(self):
        return 3.14159 * (self.radius**2)


@dataclass
class Polygon:
    points: list[Point]

    @property
    def perimeter(self):
        length = 0
        for i, point in enumerate(self.points):
            length += Line(point, self.points[(i + 1) % len(self.points)]).length

        return length


@dataclass
class LineRelationship:
    line1: Line
    line2: Line

    def __post_init__(self):
        self.dx1 = self.line1.p1.x - self.line1.p2.x
        self.dy1 = self.line1.p1.y - self.line1.p2.y

        self.dx2 = self.line2.p1.x - self.line2.p2.x
        self.dy2 = self.line2.p1.y - self.line2.p2.y

    @property
    def is_parallel(self):
        return self.dx1 * self.dy2 == self.dy1 * self.dx2

    @property
    def is_perpendicular(self):
        return self.dx1 * self.dx2 + self.dy1 * self.dy2 == 0


@dataclass
class CircleRelationShip:
    circle1: Circle
    circle2: Circle

    @property
    def is_intersect(self):
        center_distance = Line(self.circle1.center, self.circle2.center).length

        return center_distance <= self.circle1.radius + self.circle2.radius


# ========= task 2 classes =========


@dataclass
class RedCircle:
    point: Point
    moving_speed: Point
    life: int = 10

    def move(self):
        if self.life > 0:
            self.point.x += self.moving_speed.x
            self.point.y += self.moving_speed.y

    @property
    def alive(self):
        return self.life > 0

    @property
    def life_point(self):
        return max(self.life, 0)


@dataclass
class AttackCircle:
    point: Point
    attack_point: int
    attack_range: int

    def attack(self, target: RedCircle):
        if target.alive and Line(self.point, target.point).length <= self.attack_range:
            target.life -= self.attack_point


@dataclass
class BlueCircle(AttackCircle):
    attack_point: int = 1
    attack_range: int = 2


@dataclass
class GreenCircle(AttackCircle):
    attack_point: int = 2
    attack_range: int = 4


if __name__ == "__main__":
    # ========= task 1 =========
    line_a = Line(Point(-6, 1), Point(2, 4))
    line_b = Line(Point(-6, -1), Point(2, 2))
    line_c = Line(Point(-1, 6), Point(-4, -4))

    circle_a = Circle(Point(6, 3), 2)
    circle_b = Circle(Point(8, 1), 1)

    polygon_a = Polygon([Point(2, 0), Point(-1, -2), Point(4, -4), Point(5, -1)])

    print("Task 1:")
    print(
        "Are Line A and Line B parallel?",
        LineRelationship(line_a, line_b).is_parallel,
    )
    print(
        "Are Line C and Line A perpendicular?",
        LineRelationship(line_c, line_a).is_perpendicular,
    )
    print(
        "Print the area of Circle A:",
        circle_a.area,
    )
    print(
        "Are Circle A and Circle B intersect?",
        CircleRelationShip(circle_a, circle_b).is_intersect,
    )
    print(
        "Print the perimeter of Polygon A:",
        polygon_a.perimeter,
    )

    # ========= task 2 =========
    e1 = RedCircle(Point(-10, 2), Point(2, -1))
    e2 = RedCircle(Point(-8, 0), Point(3, 1))
    e3 = RedCircle(Point(-9, -1), Point(3, 0))
    e_list: list[RedCircle] = [e1, e2, e3]

    t1 = BlueCircle(Point(-3, 2))
    t2 = BlueCircle(Point(-1, -2))
    t3 = BlueCircle(Point(4, 2))
    t4 = BlueCircle(Point(7, 0))

    a1 = GreenCircle(Point(1, 1))
    a2 = GreenCircle(Point(4, -3))
    tower_list: list[BlueCircle | GreenCircle] = [t1, t2, t3, t4, a1, a2]

    print("\nTask 2:")
    for _ in range(10):
        for e in e_list:
            e.move()

        for tower in tower_list:
            for e in e_list:
                tower.attack(e)

    for i, e in enumerate(e_list, 1):
        print(f"E{i}", e.point, e.life_point)
