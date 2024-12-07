from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    
    def __str__(self):
        return self.image.name