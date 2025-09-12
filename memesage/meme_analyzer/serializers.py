from rest_framework import serializers

class MemeUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

class MemeResultSerializer(serializers.Serializer):
    extracted_text = serializers.CharField()
    tags = serializers.ListField(child=serializers.CharField())
    confidence_scores = serializers.ListField(child=serializers.FloatField())
