class Sensor {
    constructor(car) {
        this.car = car;
        this.rayCount = 5; // raycount
        this.rayLength = 150;
        this.raySpread = Math.PI / 2;
        this.rays = []; // array to store ray start, end pt
        this.readings = []; // store sensor readings
    }

    update(roadBorders, traffic) {
        this.#castRays();
        this.readings = [];
        for (let i = 0; i < this.rays.length; i++) {
            this.readings.push(
                this.#getReading(
                    this.rays[i],
                    roadBorders,
                    traffic
                )
            );
        }
    }

    #getReading(ray, roadBorders, traffic) {
        let touches = []; // store sensor reading
        for (let i = 0; i < roadBorders.length; i++) { // check for intersection with road borders
            const touch = getIntersection(
                ray[0],
                ray[1],
                roadBorders[i][0],
                roadBorders[i][1]
            );
            if (touch) {
                touches.push(touch);
            }
        }
        
        // Fixed: Added missing opening curly brace
        for (let i = 0; i < traffic.length; i++) { // check intersection with traffic cars
            const poly = traffic[i].polygon;
            for (let j = 0; j < poly.length; j++) {
                const value = getIntersection(
                    ray[0],
                    ray[1],
                    poly[j],
                    poly[(j + 1) % poly.length]
                );
                if (value) {
                    touches.push(value);
                }
            }
        }
        
        if (touches.length == 0) {
            return null; // no obstacle detected
        } else {
            const offsets = touches.map(e => e.offset);
            const minOffset = Math.min(...offsets);
            return touches.find(e => e.offset == minOffset); // return closest obstacle
        }
    }

    #castRays() {
        this.rays = [];
        for (let i = 0; i < this.rayCount; i++) {
            const rayAngle = lerp(
                this.raySpread / 2,
                -this.raySpread / 2,
                this.rayCount == 1 ? 0.5 : i / (this.rayCount - 1)
            ) + this.car.angle;
            const start = { x: this.car.x, y: this.car.y };
            const end = {
                x: this.car.x -
                    Math.sin(rayAngle) * this.rayLength,
                y: this.car.y -
                    Math.cos(rayAngle) * this.rayLength
            };
            this.rays.push([start, end]);
        }
    }

    draw(ctx) {
        for (let i = 0; i < this.rayCount; i++) {
            let end = this.rays[i][1];
            if (this.readings[i]) {
                end = this.readings[i]; // if obstacle detected, use it as end pt
            }
            
            // Draw ray from car to obstacle
            ctx.beginPath();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(255, 255, 0, 0.5)";
            ctx.moveTo(
                this.rays[i][0].x,
                this.rays[i][0].y
            );
            ctx.lineTo(
                end.x,
                end.y
            );
            ctx.stroke();
            
            // Draw hit point (remaining part of ray)
            if (this.readings[i]) {
                ctx.beginPath();
                ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
                ctx.fillStyle = "red";
                ctx.fill();
            }
        }
    }
}

// Helper functions needed for the Sensor class to work:

// Linear interpolation function
function lerp(A, B, t) {
    return A + (B - A) * t;
}

// Function to check if two line segments intersect
function getIntersection(A, B, C, D) {
    const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
    const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
    const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);

    if (bottom != 0) {
        const t = tTop / bottom;
        const u = uTop / bottom;
        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            return {
                x: A.x + t * (B.x - A.x),
                y: A.y + t * (B.y - A.y),
                offset: t
            };
        }
    }

    return null;
}
