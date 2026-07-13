char command;

void setup()
{
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);

  Serial.begin(9600);
}

void loop()
{
  if (Serial.available())
  {
    command = Serial.read();

    if(command == 'F')
    {
      forward();
    }

    else if(command == 'L')
    {
      left();
    }

    else if(command == 'R')
    {
      right();
    }

    else if(command == 'S')
    {
      stopRobot();
    }
  }
}

void forward()
{
  digitalWrite(8,HIGH);
  digitalWrite(9,LOW);

  digitalWrite(10,HIGH);
  digitalWrite(11,LOW);
}

void left()
{
  digitalWrite(8,LOW);
  digitalWrite(9,HIGH);

  digitalWrite(10,HIGH);
  digitalWrite(11,LOW);
}

void right()
{
  digitalWrite(8,HIGH);
  digitalWrite(9,LOW);

  digitalWrite(10,LOW);
  digitalWrite(11,HIGH);
}

void stopRobot()
{
  digitalWrite(8,LOW);
  digitalWrite(9,LOW);

  digitalWrite(10,LOW);
  digitalWrite(11,LOW);
}